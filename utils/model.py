import math

import torch
import torch.nn as nn


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

        self.head_linear = nn.Linear(
            linear_input_size,
            num_classes
        )

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

                    nn.init.normal_(
                        m.bias,
                        -bound,
                        bound
                    )

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
    n_epochs, optimizer, model, loss_fn, 
    train_loader, val_loader, device, start_epoch=1, scores=None
    ):
    #start = time.perf_counter()
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
        # Проходим в цикле по нашему набору данных по батчам, создаваемым загрузчиком данных
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
            optimizer.zero_grad()   # Избавившись от градиентов с предыдущей итерации
            loss.backward()         # выполняем обратный проход
            optimizer.step()        # обновляем модель
            
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
                loss = loss_fn(outputs, labels)  # <5>вычисляем минимизируемую функцию потерь
                
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