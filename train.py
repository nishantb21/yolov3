from core.config import cfg

def main(config, training_mode):
    if training_mode == "BACKBONE":
        from core.builders import ImageNetBuilder
        builder_obj = ImageNetBuilder(config)
        builder_obj.build()
        trainer_obj = builder_obj.get_trainer()
        trainer_obj.train() 
    
    else:
        raise NotImplementedError("Training mode not implemented")


if __name__ == "__main__":
    main(cfg, "BACKBONE")