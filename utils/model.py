import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LunaModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        conv_channels=8,
        input_shape=(32, 48, 48),
        num_classes=2
    ):
        super().__init__()

        self.input_shape = tuple(input_shape)

        self.tail_batchnorm = nn.BatchNorm3d(in_channels)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels,     conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8
        )

        # Four MaxPool3d(2, 2) operations reduce each
        # spatial dimension by a factor of 2^4 = 16.
        final_shape = tuple(
            dimension // 16
            for dimension in self.input_shape
        )

        final_channels = conv_channels * 8

        linear_input_size = (
            final_channels
            * final_shape[0]
            * final_shape[1]
            * final_shape[2]
        )

        self.head_linear = nn.Linear(linear_input_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d
            }:
                nn.init.kaiming_normal_(
                    m.weight.data,
                    a=0,
                    mode='fan_out',
                    nonlinearity='relu'
                )

                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)

                    bound = 1 / math.sqrt(fan_out)

                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(block_out.size(0), -1)

        logits = self.head_linear(conv_flat)

        return logits


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            conv_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)
    
    
def training_loop(
    n_epochs:int, optimizer, model, loss_fn, 
    train_loader, val_loader, device, start_epoch=1, scores=None
    )->dict:
    """Pytorch training loop with validation

    Args:
        n_epochs (int): Number of epochs
        optimizer (_type_): an optimizer function
        model (_type_): Pytorch model
        loss_fn (_type_): loss function
        train_loader (_type_): data loader class for the train subset
        val_loader (_type_):  data loader class for the validation subset
        device (_type_): Device to load the model. Either Cpu or GPU
        start_epoch (int, optional): Initial epoch. Defaults to 1.
        scores (dict, optional): Dictionary to store training results. Defaults to None.

    Returns:
        dict: _description_
    """    

    if scores is None:
        scores = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    for epoch in range(start_epoch, n_epochs + 1):  
        model.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # iterate over batches
        for imgs, labels, _, _ in train_loader: 
            
            imgs, labels = imgs.to(device), labels.to(device)
            # print(
            #     "imgs:", imgs.device,
            #     "labels:", labels.device,
            #     "model:", next(model.parameters()).device
            # )

            # outputs = model(imgs)

            # print("outputs:", outputs.device)
            
            # Forward pass and loss calc
            outputs = model(imgs)
            loss = loss_fn(outputs, labels) 
            
            # Backward
            optimizer.zero_grad()   # purge grads from previous iter
            loss.backward()         # backward pass
            optimizer.step()        # upd model
            
            # Metrics
            batch_size = labels.size(0)

            train_loss += loss.item() * batch_size

            predicted = outputs.argmax(dim=1)

            train_correct += (predicted == labels).sum().item()

            train_total += batch_size
            
        mean_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels, _, _ in val_loader:
                
                imgs, labels = imgs.to(device), labels.to(device)
                
                # Forward pass and loss calc
                outputs = model(imgs) 
                loss = loss_fn(outputs, labels)  # compute loss func
                
                # Accuracy Metrics
                batch_size = labels.size(0)

                val_loss += loss.item() * batch_size

                predicted = outputs.argmax(dim=1)

                val_correct += (predicted == labels).sum().item()

                val_total += batch_size
        
        mean_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # save metrics 
        scores['train_loss'].append(mean_train_loss)
        scores['train_acc'].append(train_acc)
        
        scores['val_loss'].append(mean_val_loss)
        scores['val_acc'].append(val_acc)
        
        #if epoch == 1 or epoch % 25 == 0:
        print(f"Epoch: {epoch}, "
            f"Train loss: {mean_train_loss:.3g}, Trn Acc : {train_acc:.3g}, " 
            f"Valid loss:   {mean_val_loss:.3g}, Val Acc : {val_acc:.3g} "
        )
        
    return scores

def evaluate_model(model, test_loader, loss_fn, device):
    """evaluating function"""    
    
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for imgs, labels, _, _ in test_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(imgs)

            # Loss
            loss = loss_fn(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            # Probabilities
            probabilities = torch.softmax(logits, dim=1)

            positive_probabilities = probabilities[:, 1]

            # Predictions
            predictions = logits.argmax(dim=1)

            total_correct += (predictions == labels).sum().item()

            total_samples += batch_size

            # Move results to CPU
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_probabilities.append(positive_probabilities.cpu())

    mean_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_predictions).numpy()
    y_prob = torch.cat(all_probabilities).numpy()

    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

class Augmentation3D(nn.Module):
    def __init__(
            self,
            flip=False,
            offset:float=None,
            scale:float=None,
            rotate=None,
            noise=None
    ):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g):
        """
        Apply random spatial transformations and Gaussian noise to a CT patch. 
     
        Expected input (in HU):
            input_g : (N, C, I, R, C)

        Returns (in HU):
            augmented_input_g (N, C, I, R, C)
        """

        transform_t = self._build3dTransformMatrix(input_g.device)

        transform_t = transform_t.expand(input_g.shape[0], -1, -1)

        #transform_t = transform_t.to(input_g.device, torch.float32)

        # Generate a 3-D sampling grid.
        # affine_grid expects a 3 x 4 matrix for 5-D input.
        affine_t = F.affine_grid(
            transform_t[:, :3],
            input_g.size(),
            align_corners=False
        )
         
        # CT image:
        # trilinear interpolation is appropriate for continuous HU values.
        augmented_input_g = F.grid_sample(
            input_g,
            affine_t,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )

        # Add Gaussian noise only to the CT image.
        if self.noise is not None:
            noise_t = torch.randn_like(augmented_input_g) * float(self.noise)

            augmented_input_g += noise_t

        # clip
        augmented_input_g = torch.clamp(augmented_input_g, -1000.0, 1000.0)

        return augmented_input_g

    def _build3dTransformMatrix(self, device):

        transform_t = torch.eye(4, device=device, dtype=torch.float32)

        # Random reflection.
        if self.flip:
            for i in range(3):
                if torch.rand((), device=device) > 0.5:
                    transform_t[i, i] *= -1
        
        # Random translation.
        if self.offset is not None:
            offset_float = float(self.offset)
            
            # I = 0, R = 1, C = 2
            for i in range(3):
                random_float = torch.rand((), device=device) * 2 - 1
                transform_t[i, 3] =  offset_float * random_float

            # Random scaling.
            if self.scale is not None:
                scale_float = float(self.scale)
                for i in range(3):
                    random_float = torch.rand((), device=device) * 2 - 1
                    transform_t[i, i] *= 1.0 + scale_float * random_float
                

        # Random rotation in R-C plane around the I axis.
        # (not three because voxels are not cubic)
        if self.rotate is not None:
            max_angle = self.rotate
            
            if isinstance(max_angle, bool):
                max_angle = torch.pi
            else:
                max_angle = float(max_angle)

            # Random angle in [-max_angle, +max_angle].
            angle_rad = (torch.rand((), device=device) * 2.0 - 1.0) * max_angle
                       
            s = torch.sin(angle_rad)
            c = torch.cos(angle_rad)

            rotation_t = torch.tensor([
                [1, 0, 0, 0],
                [0, c, -s, 0],
                [0, s,  c, 0],
                [0, 0, 0, 1]], device=device, dtype=torch.float32)

            transform_t @= rotation_t

        return transform_t
    
    
class LunaModel2(nn.Module):

    def __init__(
        self,
        in_channels=1,
        conv_channels=8,
        input_shape=(32, 48, 48),
        num_classes=2,
    ):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv3d(
                in_channels,
                conv_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(conv_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),

            # Block 2
            nn.Conv3d(
                conv_channels,
                conv_channels * 2,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(conv_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),

            # Block 3
            nn.Conv3d(
                conv_channels * 2,
                conv_channels * 4,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(conv_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )

        # Determine the flattened feature size automatically.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_shape)

            feature_shape = self.features(dummy).shape[1:]
            num_features = int(torch.tensor(feature_shape).prod())

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x