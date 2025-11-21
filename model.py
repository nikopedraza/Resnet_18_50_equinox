import jax
from jax import numpy as jnp
from jax import random as jr
import equinox as eqx
from typing import Union

class Conv7x7(eqx.Module):
    conv:eqx.nn.Conv2d

    def __init__(self,in_channels,out_channels,key):
        self.conv = eqx.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3,key=key)
    def __call__(self,x):
        return self.conv(x)

class Conv3x3(eqx.Module):
    conv:eqx.nn.Conv2d

    def __init__(self,in_channels,out_channels,downsample,key):
        self.conv = eqx.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=downsample, padding=1,key=key)
    def __call__(self,x):
        return self.conv(x)

class Conv_Norm(eqx.Module):
    block:eqx.nn.Conv2d
    bn :eqx.nn.BatchNorm

    def __init__(self,block,bn_channels):
        self.block = block
        self.bn = eqx.nn.BatchNorm(bn_channels,axis_name="batch",mode="batch")

    def __call__(self,x,state):
        x = self.block(x)
        x,state = self.bn(x,state)
        return x,state


#Residual Blocks
class ResBasicBlock(eqx.Module):
    conv1: Conv_Norm
    conv2: Conv_Norm
    shortcut: Conv_Norm
    in_channels: int
    out_channels: int
    downsample: int

    def __init__(self, in_channels, out_channels, downsample, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        k1, k2, k3, _ = jax.random.split(key, 4)

        # Main path
        c1 = Conv3x3(in_channels, out_channels, downsample, k1)
        c2 = Conv3x3(out_channels, out_channels, 1,        k2)

        self.conv1 = Conv_Norm(c1, out_channels)
        self.conv2 = Conv_Norm(c2, out_channels)


        c3 = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=downsample,
            padding=0,
            key=k3,
        )
        self.shortcut = Conv_Norm(c3, out_channels)

    def __call__(self, x, state):
        residual = x

        x, state = self.conv1(x, state)
        x = jax.nn.relu(x)
        x, state = self.conv2(x, state)

        # jax.debug.print("Block in channels -> {x}", x=self.in_channels)
        # jax.debug.print("Block out channels -> {x}", x=self.out_channels)
        # jax.debug.print("downsample -> {x}", x=self.downsample)

        # Use shortcut if shape would differ
        if self.in_channels != self.out_channels or self.downsample == 2:
            # jax.debug.print("here")
            residual, state = self.shortcut(residual, state)

        x = x + residual
        return jax.nn.relu(x), state


class ResBottleNeckBlock(eqx.Module):
    conv1: Conv_Norm
    conv2: Conv_Norm
    conv3: Conv_Norm
    shortcut:Conv_Norm
    in_channels:int
    out_channels:int
    downsample:int

    def __init__(self,in_channels,out_channels,downsample,key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        keys = jr.split(key,3)
        c1 = eqx.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=downsample,key=keys[0])
        c2 = Conv3x3(out_channels,out_channels,1,keys[1])
        c3 = eqx.nn.Conv2d(out_channels,out_channels*4,kernel_size=1,stride=1,key=keys[2])
        c4 = eqx.nn.Conv2d(in_channels,out_channels*4,kernel_size=1,stride=downsample,key=keys[3])
        self.conv1 = Conv_Norm(c1,out_channels)
        self.conv2 = Conv_Norm(c2,out_channels)
        self.conv3 = Conv_Norm(c3,out_channels*4)
        self.shortcut = Conv_Norm(c4,out_channels*4)


    def __call__(self,x,state):
        residual = x
        x,state = self.conv1(x,state)
        x = jax.nn.relu(x)
        x,state = self.conv2(x,state)
        x = jax.nn.relu(x)
        x,state = self.conv3(x,state)
        # jax.debug.print("in_channel -> {x}",x=self.in_channels)
        # jax.debug.print("out_channel -> {x}",x=self.out_channels)
        # jax.debug.print("downsample -> {x}",x=self.downsample)
        if self.in_channels != self.out_channels*4 or self.downsample==2:
            residual,state = self.shortcut(residual,state)

        x = x + residual

        return jax.nn.relu(x),state



#Non Residual Blocks
class BasicBlock(eqx.Module):
    conv1: Conv_Norm
    conv2: Conv_Norm
    shortcut:Conv_Norm
    in_channels:int
    out_channels:int
    downsample:int

    def __init__(self,in_channels,out_channels,downsample,key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        keys = jax.random.split(key, 3)
        c1 = Conv3x3(in_channels,out_channels,downsample,keys[0])
        c2 = Conv3x3(out_channels,out_channels,1,keys[1])
        c3 = eqx.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=downsample,key=keys[2])
        self.conv1 = Conv_Norm(c1,out_channels)
        self.conv2 = Conv_Norm(c2,out_channels)
        self.shortcut = Conv_Norm(c3,out_channels)

    def __call__(self,x,state):
        x,state = self.conv1(x,state)
        x = jax.nn.relu(x)
        x,state = self.conv2(x,state)
        return jax.nn.relu(x),state

class BottleNeckBlock(eqx.Module):
    conv1: Conv_Norm
    conv2: Conv_Norm
    conv3: Conv_Norm
    shortcut:Conv_Norm
    in_channels:int
    out_channels:int
    downsample:int

    def __init__(self,in_channels,out_channels,downsample,key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        keys = jr.split(key,4)
        c1 = eqx.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=downsample,key=keys[0])
        c2 = Conv3x3(out_channels,out_channels,1,keys[1])
        c3 = eqx.nn.Conv2d(out_channels,out_channels*4,kernel_size=1,stride=1,key=keys[2])
        c4 = eqx.nn.Conv2d(in_channels,out_channels*4,kernel_size=1,stride=downsample,key=keys[3])
        self.conv1 = Conv_Norm(c1,out_channels)
        self.conv2 = Conv_Norm(c2,out_channels)
        self.conv3 = Conv_Norm(c3,out_channels*4)
        self.shortcut = Conv_Norm(c4,out_channels*4)

    def __call__(self,x,state):
        x,state = self.conv1(x,state)
        x = jax.nn.relu(x)
        x,state = self.conv2(x,state)
        x = jax.nn.relu(x)
        x,state = self.conv3(x,state)
        return jax.nn.relu(x),state

class ResNetLayer(eqx.Module):
    block: Union[ResBottleNeckBlock,ResBasicBlock,BottleNeckBlock,BasicBlock]
    layer:tuple
    def __init__(self,in_channels,out_channels,block,key,n=1):
        self.block = block
        block_expansion = 1 if issubclass(block,(ResBasicBlock,BasicBlock))  else 4 #bottleneck has expansion = 4
        downsample = 2 if in_channels != out_channels*block_expansion else 1
        keys = jr.split(key,n)

        blocks = [block(in_channels,out_channels,downsample,keys[0])]

        for i in range(1,n):
            blocks.append(block(out_channels*block_expansion,out_channels,1,keys[i]))
        self.layer = tuple(blocks)

    def __call__(self, x,state):
        for blk in self.layer:
            x, state = blk(x, state)
        return x,state



class ResNet(eqx.Module):
    input_size:int
    num_classes:int
    layer_size:tuple
    layers:tuple
    block: Union[ResBasicBlock,ResBottleNeckBlock,BottleNeckBlock,BasicBlock]
    maxpool: eqx.nn.MaxPool2d
    avgpool: eqx.nn.AdaptiveAvgPool2d
    conv1: Conv7x7
    fc: eqx.nn.Linear



    base_channel: int = 64


    def __init__(self,input_size=3,num_classes=200,layer_size=(1,1,1,1),block=ResBasicBlock,key=jax.random.key(0)):
        self.input_size = input_size
        self.num_classes = num_classes
        self.layer_size = layer_size
        self.block = block
        keys = jr.split(key,6)
        block_expansion = 1 if (issubclass(block, ResBasicBlock) or issubclass(block, BasicBlock)) else 4
        self.conv1 = Conv7x7(input_size,self.base_channel,keys[0])
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=(3,3),stride=(2,2),padding=1)

        in_channels = self.base_channel
        out_channels = self.base_channel

        layers= [ResNetLayer(in_channels,out_channels,block,keys[1],layer_size[0])]

        for ii in range(1,len(layer_size)):
            in_channels = out_channels*block_expansion
            out_channels = 2*out_channels

            layers.append(ResNetLayer(in_channels,out_channels,block,keys[ii+1],layer_size[ii]))


        self.layers = tuple(layers)

        self.avgpool = eqx.nn.AdaptiveAvgPool2d(target_shape=(1,1))
        self.fc = eqx.nn.Linear(out_channels*block_expansion,num_classes,key=keys[-1])

    def __call__(self, x,state):
        x = self.conv1(x)
        x = self.maxpool(x)

        for i,layer in enumerate(self.layers):
            x,state = layer(x,state)



        x = self.avgpool(x)
        x = jnp.ravel(x)
        x = self.fc(x)

        return x,state



