import hydra
import os
import numpy as np

from utils import generate_run_name_hydra, remove_target_key, set_seed_all, get_device_from_config
from datasets import TaskShuffledLabels
from lr_scheduler import LRSchedulerWrapper
from network import NetworkPropagator


@hydra.main(version_base=None, config_path="config/", config_name="main.yaml")
def main(config):
    ########################
    # Task and model setup #
    ########################
    if config.make_deterministic: set_seed_all(seed=config.seed)

    learning_scenario = hydra.utils.instantiate(config=config.learning_scenario)

    # Shuffle the task order. Note that the tasks remain the same, but the order of learning changes.
    if config.use_task_permutation:
        task_permutation = np.random.permutation(config.n_tasks).tolist()
        print(task_permutation)
        target_transform = TaskShuffledLabels(num_tasks=config.n_tasks,
                                              num_classes_per_task=config.n_classes_per_task,
                                              num_outputs=learning_scenario.get_output_dim(),
                                              task_permutation=task_permutation)
    else:
        task_permutation = None
        target_transform = None

    dataset = hydra.utils.instantiate(config=config.dataset, 
                                      task_permutation=task_permutation, 
                                      target_transform=target_transform)

    # Joint training trains the network on all the data at the same time. Testing still uses separate tasks.
    if config.is_joint_training:
        train_data_loader, test_data_loader = dataset.get_joint_data_loaders(batch_size=config.batch_size, 
                                                                             shuffle=config.shuffle_loader_data, 
                                                                             num_workers=config.n_workers,)
    else:
        train_data_loader, test_data_loader = dataset.get_data_loaders(batch_size=config.batch_size, 
                                                                       shuffle=config.shuffle_loader_data, 
                                                                       num_workers=config.n_workers,)
    
    callbacks = [hydra.utils.instantiate(config.callbacks[callback_name]) for callback_name in config.callbacks]

    model = hydra.utils.instantiate(config=config.model, output_dim=learning_scenario.get_output_dim())

    optimizer = hydra.utils.instantiate(config=config.optimizer, params=model.parameters())

    loss_fn = hydra.utils.instantiate(config=config.loss)

    network_propagator = NetworkPropagator(model=model,
                                           configuration=remove_target_key(config),
                                           learning_scenario=learning_scenario,
                                           device=get_device_from_config(configuration=config))
    
    loss_entity = hydra.utils.instantiate(config=config.loss_entity,
                                          model=model,
                                          loss_function=loss_fn,
                                          optimizer=optimizer,
                                          configuration=remove_target_key(config),
                                          data_loaders=train_data_loader,
                                          network_propagator=network_propagator)

    trainer = hydra.utils.instantiate(config=config.trainer, 
                                      model=model, 
                                      loss_entity=loss_entity,  
                                      callbacks=callbacks, 
                                      configuration=remove_target_key(config),
                                      network_propagator=network_propagator)
    
    scheduler_wrapper = LRSchedulerWrapper()

    print(generate_run_name_hydra(config))

    #############
    # Task Loop #
    #############
    for task_idx in [0] if config.is_joint_training else range(config.n_tasks):

        # signal the start of a task
        trainer.prepare_task(task_id=task_idx)
        consolidate_importances = getattr(optimizer, "consolidate_importances", None)  # for custom optimizers
        if callable(consolidate_importances):
            optimizer.consolidate_importances()
        
        # prepare the learning rate (LR) scheduler
        if config.use_lr_scheduler:
            optimizer.param_groups[0]['lr'] = config.learning_rate  # reset the learning rate before training a new task
            lr_scheduler = hydra.utils.instantiate(config=config.lr_scheduler, optimizer=optimizer)
            scheduler_wrapper.set_scheduler(lr_scheduler)
        
        ############
        # Training #
        ############
        for epoch in range(config.epochs):

            print(f"\nTask {task_idx + 1}, training epoch > {epoch + 1} / {config.epochs} <")
            _, loss = trainer.train_task(task_id=task_idx, 
                                         data_loader=train_data_loader[task_idx])
   
            if config.test_after_epoch:
                trainer.test_task(task_id=task_idx, 
                                  data_loader=test_data_loader[task_idx], 
                                  enable_logging=False)
                
            # Update the loss scheduler
            if config.use_lr_scheduler:
                scheduler_wrapper.step(loss=loss)
        
        ###########
        # Testing #
        ###########
        for index, data_loader in enumerate(test_data_loader):
            print(f"\nTask {index + 1}, testing")
            trainer.test_task(task_id=index,
                              data_loader=data_loader,
                              enable_logging=True)
            
            # only check task performance of trained tasks
            if index >= task_idx and not config.is_joint_training:
                break

    # Call the routine one more time to compute quantities such as importance scores which might be relevant for logging
    trainer.prepare_task(task_id=task_idx)


if __name__ == '__main__':
    os.environ["HYDRA_FULL_ERROR"] = "1"  # enables chained exceptions for hydra errors
    main()
