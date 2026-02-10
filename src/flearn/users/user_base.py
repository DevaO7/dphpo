import torch
import copy
from opacus import PrivacyEngine
from torch.utils.data import DataLoader

class User: 
    def __init__(self, model, train_loader, test_loader, loss_fn_name, use_cuda, local_updates, dp, optimizer, noise_multiplier, max_grad_norm, id, sample_rate, x_label, y_label):
        self.use_cuda = use_cuda
        self.traindataset = train_loader.dataset
        self.train_samples = len(self.traindataset)
        self.batch_size = max(1, int(sample_rate * len(self.traindataset)))
        self.trainloaderfull = train_loader
        loss_fn = getattr(torch.nn, loss_fn_name)(reduction='mean')
        self.loss = getattr(torch.nn, loss_fn_name)(reduction='mean')
        if dp:
            train_loader = DataLoader(self.traindataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
            self.privacy_engine = PrivacyEngine()
            generator = torch.Generator(device='cuda' if use_cuda else 'cpu').manual_seed(id)
            self.model, self.optimizer, self.dp_loss, self.dp_train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                noise_generator=generator, 
                criterion=loss_fn, 
                grad_sample_mode="ghost", 
                loss_reduction="mean"
            )
        self.delta_model = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.server_model = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.testloaderfull = test_loader
        self.local_updates = local_updates
        self.x_label = x_label
        self.y_label = y_label
        
    def set_parameters(self, server_model):
        for old_param, new_param, local_param, server_param in zip(self.model.parameters(), server_model.parameters(),
                                                                self.local_model, self.server_model):
            old_param.data = new_param.data.clone().to(old_param.data.device)
            local_param.data = new_param.data.clone().to(local_param.data.device)
            server_param.data = new_param.data.clone().to(server_param.data.device)
            if (new_param.grad != None):
                if (old_param.grad == None):
                    old_param.grad = torch.zeros_like(new_param.grad)

                if (local_param.grad == None):
                    local_param.grad = torch.zeros_like(new_param.grad)

                old_param.grad.data = new_param.grad.data.clone().to(old_param.grad.data.device)
                local_param.grad.data = new_param.grad.data.clone().to(local_param.grad.data.device)
        # self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def test_error_and_loss(self):
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.eval()
        correct = 0
        loss_sum = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.testloaderfull:
                x = batch[self.x_label]
                y = batch[self.y_label]
                if self.use_cuda:
                    x, y = x.cuda(), y.cuda()
                out = self.model(x)
                correct += (out.argmax(dim=1) == y).sum().item()
                bs = y.size(0)
                loss_sum += self.loss(out, y).item() * bs
                n += bs
        self.model.cpu()
        return correct, (loss_sum / n), n

    def train_error_and_loss(self):
        """Returns metrics evaluated on train data."""
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.eval()
        train_acc = 0
        total_loss = 0
        n = 0
        with torch.no_grad():
            for batch in self.trainloaderfull:
                x = batch[self.x_label]
                y = batch[self.y_label]
                if self.use_cuda:
                    x, y = x.cuda(), y.cuda()
                output = self.model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                bs = y.size(0)
                total_loss += self.loss(output, y).item()*bs
                n += bs
        avg_loss = total_loss / n
        self.model.cpu()
        return train_acc, avg_loss, n