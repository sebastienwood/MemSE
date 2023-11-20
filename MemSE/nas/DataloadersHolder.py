__all__ = ["DataloadersHolder"]

class DataloadersHolder:
    def __init__(self, run_manager) -> None:
        self.run_manager = run_manager
        self.holder = {}
        
    def get_image_size(self, img_size: int):
        if img_size not in self.holder:
            self.run_manager._loader.assign_active_img_size(img_size)
            data_loader = self.run_manager._loader.build_sub_train_loader(
                    n_images=2000, batch_size=200
                )
            val_dataset = []
            for batch in self.run_manager.valid_loader:
                if isinstance(batch, dict):
                    images, labels = batch['image'], batch['label']
                else:
                    images, labels = batch
                val_dataset.append((images, labels))
            self.holder[img_size] = (data_loader, val_dataset)
        return self.holder[img_size]