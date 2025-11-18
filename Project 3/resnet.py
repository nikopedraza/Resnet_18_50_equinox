from typing import Callable, Optional, Sequence, Tuple, Type

import equinox as eqx
import jax
import jax.numpy as jnp


def _conv3x3(in_planes: int, out_planes: int, stride: int, *, key: jax.Array) -> eqx.nn.Conv2d:
    """3x3 convolution with padding."""
    return eqx.nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        use_bias=False,
        key=key,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int, *, key: jax.Array) -> eqx.nn.Conv2d:
    """1x1 convolution used for channel matching in shortcuts."""
    return eqx.nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        use_bias=False,
        key=key,
    )


class Downsample(eqx.Module):
    conv: eqx.nn.Conv2d
    bn: eqx.nn.BatchNorm

    def __init__(self, inplanes: int, planes: int, stride: int, *, key: jax.Array):
        self.conv = _conv1x1(inplanes, planes, stride, key=key)
        self.bn = eqx.nn.BatchNorm(
            planes,
            axis=(0, 2, 3),
            use_bias=True,
            use_scale=True,
            momentum=0.9,
            eps=1e-5,
        )

    def __call__(self, x: jnp.ndarray, *, inference: bool = False) -> jnp.ndarray:
        x = self.conv(x)
        x = self.bn(x, inference=inference)
        return x


class BasicBlock(eqx.Module):
    """ResNet basic residual block used by ResNet-18/34."""

    expansion: int = 1

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    downsample: Optional[Downsample]
    stride: int

    def __init__(
        self,
        inplanes: int,
        planes: int,
        *,
        stride: int = 1,
        downsample: Optional[Downsample] = None,
        key: jax.Array,
    ):
        k1, k2 = jax.random.split(key, 2)

        self.conv1 = _conv3x3(inplanes, planes, stride, key=k1)
        self.bn1 = eqx.nn.BatchNorm(
            planes,
            axis=(0, 2, 3),
            use_bias=True,
            use_scale=True,
            momentum=0.9,
            eps=1e-5,
        )
        self.conv2 = _conv3x3(planes, planes * self.expansion, 1, key=k2)
        self.bn2 = eqx.nn.BatchNorm(
            planes * self.expansion,
            axis=(0, 2, 3),
            use_bias=True,
            use_scale=True,
            momentum=0.9,
            eps=1e-5,
        )
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: jnp.ndarray, *, inference: bool = False) -> jnp.ndarray:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, inference=inference)
        out = jax.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, inference=inference)

        if self.downsample is not None:
            identity = self.downsample(x, inference=inference)

        out = out + identity
        out = jax.nn.relu(out)
        return out


class Bottleneck(eqx.Module):
    """ResNet bottleneck block used by ResNet-50/101/152."""

    expansion: int = 4

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    conv3: eqx.nn.Conv2d
    bn3: eqx.nn.BatchNorm
    downsample: Optional[Downsample]
    stride: int

    def __init__(
        self,
        inplanes: int,
        planes: int,
        *,
        stride: int = 1,
        downsample: Optional[Downsample] = None,
        key: jax.Array,
    ):
        k1, k2, k3 = jax.random.split(key, 3)

        self.conv1 = _conv1x1(inplanes, planes, 1, key=k1)
        self.bn1 = eqx.nn.BatchNorm(
            planes,
            axis=(0, 2, 3),
            use_bias=True,
            use_scale=True,
            momentum=0.9,
            eps=1e-5,
        )
        self.conv2 = _conv3x3(planes, planes, stride, key=k2)
        self.bn2 = eqx.nn.BatchNorm(
            planes,
            axis=(0, 2, 3),
            use_bias=True,
            use_scale=True,
            momentum=0.9,
            eps=1e-5,
        )
        self.conv3 = _conv1x1(planes, planes * self.expansion, 1, key=k3)
        self.bn3 = eqx.nn.BatchNorm(
            planes * self.expansion,
            axis=(0, 2, 3),
            use_bias=True,
            use_scale=True,
            momentum=0.9,
            eps=1e-5,
        )
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: jnp.ndarray, *, inference: bool = False) -> jnp.ndarray:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, inference=inference)
        out = jax.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, inference=inference)
        out = jax.nn.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, inference=inference)

        if self.downsample is not None:
            identity = self.downsample(x, inference=inference)

        out = out + identity
        out = jax.nn.relu(out)
        return out


def _make_layer(
    block: Type[eqx.Module],
    inplanes: int,
    planes: int,
    blocks: int,
    *,
    stride: int,
    key: jax.Array,
) -> Tuple[eqx.nn.Sequential, int]:
    keys = jax.random.split(key, blocks + 1)
    downsample: Optional[Downsample] = None

    if stride != 1 or inplanes != planes * block.expansion:
        downsample = Downsample(inplanes, planes * block.expansion, stride, key=keys[0])

    layers = [
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            key=keys[1],
        )
    ]

    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                downsample=None,
                key=keys[i + 1],
            )
        )

    return eqx.nn.Sequential(tuple(layers)), inplanes


class ResNet(eqx.Module):
    """Configurable ResNet backbone with a classification head."""

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    relu: Callable[[jnp.ndarray], jnp.ndarray]
    maxpool: eqx.nn.MaxPool2d
    layer1: eqx.nn.Sequential
    layer2: eqx.nn.Sequential
    layer3: eqx.nn.Sequential
    layer4: eqx.nn.Sequential
    fc: eqx.nn.Linear

    def __init__(
        self,
        block: Type[eqx.Module],
        layers: Sequence[int],
        num_classes: int = 1000,
        *,
        key: jax.Array,
    ):
        if len(layers) != 4:
            raise ValueError("Expected four layer depths (e.g., [2, 2, 2, 2] for ResNet-18)")

        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

        self.conv1 = eqx.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, use_bias=False, key=k1)
        self.bn1 = eqx.nn.BatchNorm(64, axis=(0, 2, 3), use_bias=True, use_scale=True, momentum=0.9, eps=1e-5)
        self.relu = jax.nn.relu
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1, inplanes = _make_layer(block, 64, 64, layers[0], stride=1, key=k2)
        self.layer2, inplanes = _make_layer(block, inplanes, 128, layers[1], stride=2, key=k3)
        self.layer3, inplanes = _make_layer(block, inplanes, 256, layers[2], stride=2, key=k4)
        self.layer4, inplanes = _make_layer(block, inplanes, 512, layers[3], stride=2, key=k5)

        self.fc = eqx.nn.Linear(512 * block.expansion, num_classes, key=k6)

    def __call__(self, x: jnp.ndarray, *, inference: bool = False) -> jnp.ndarray:
        x = self.conv1(x)
        x = self.bn1(x, inference=inference)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, inference=inference)
        x = self.layer2(x, inference=inference)
        x = self.layer3(x, inference=inference)
        x = self.layer4(x, inference=inference)

        x = jnp.mean(x, axis=(2, 3))
        x = self.fc(x)
        return x


# Factory helpers

def resnet18(*, num_classes: int = 1000, key: jax.Array) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, key=key)


def resnet34(*, num_classes: int = 1000, key: jax.Array) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, key=key)


def resnet50(*, num_classes: int = 1000, key: jax.Array) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, key=key)


def resnet101(*, num_classes: int = 1000, key: jax.Array) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, key=key)


def resnet152(*, num_classes: int = 1000, key: jax.Array) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, key=key)
