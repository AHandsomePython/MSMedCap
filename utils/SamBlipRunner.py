from utils.runner import *
from utils.data import build_dataloader
from tqdm import tqdm

class SamBlipRunner(RunnerBase):
    def __init__(
        self,
        model,
        cfg,
    ):
        config = self.build_config(cfg)
        optimizer = self.build_optimizer(model, config)
        dataloader = build_dataloader(config)
        max_epoch = config["run"]["max_epoch"]
        device = config["run"]["device"]
        super().__init__(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            max_epoch=max_epoch,
            device=device,
        )
        self.config = config
    

    def train_step(self, samples):
        clip_shape = samples[0].shape
        sam_shape = samples[1].shape
        
        # print(samples[1].view(clip_shape[0], clip_shape[1]*clip_shape[2], clip_shape[3]).shape)
        my_samples = {
            'sam_features': samples[1].view(sam_shape[0], sam_shape[1]*sam_shape[2], sam_shape[3]).to(self.device),
            'clip_features': samples[0].view(clip_shape[0], clip_shape[1]*clip_shape[2], clip_shape[3]).to(self.device),
            'text_input': samples[2],
        }
        
        output = self.model(my_samples)
        return output['loss']

    def train_epoch(self):
        for samples in tqdm(self.dataloader):
            with torch.cuda.amp.autocast(enabled=True):
                loss = self.train_step(samples)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

    def epoch_start_hook(self, info):
        pass

    def epoch_end_hook(self, info):
        torch.save({
            'epoch': info['cur_epoch'],  # 假设你训练了5个epochs
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f"./checkpoints/sam_blip_checkpoint_{info['cur_epoch']}.pth")
        print(info)

    def build_config(self, cfg):
        with open(cfg, 'r') as file:
            _config = yaml.load(file, Loader=yaml.FullLoader)
        return _config
    
    @classmethod
    def build_optimizer(self, model, config):
        lr_scale = config["run"]["lr_layer_decay"]
        weight_decay = config["run"]["weight_decay"]
        optim_params = model.get_optimizer_params(weight_decay, lr_scale)
        # optim_params = self.model.Parameters()

        num_parameters = 0
        for p_group in optim_params:
            for p in p_group["params"]:
                num_parameters += p.data.nelement()    
        logging.info("number of trainable parameters: {}".format(num_parameters))      
                
        beta2 = config["run"]["beta2"]

        _optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(config["run"]["init_lr"]),
            betas=(0.9, beta2),
        )    
        return _optimizer