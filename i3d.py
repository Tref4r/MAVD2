import torch
import torch.nn as nn
import torch.nn.functional as F

class Unit3D(nn.Module):
    def __init__(self, in_channels, output_channels,
                 kernel_size=(1, 1, 1), stride=(1, 1, 1),
                 activation_fn=F.relu,
                 use_batch_norm=True, use_bias=False, name='unit_3d'):
        super(Unit3D, self).__init__()
        self._output_channels = output_channels
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm

        self.conv3d = nn.Conv3d(
            in_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=tuple(k // 2 for k in kernel_size),
            bias=use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(output_channels, eps=1e-3, momentum=0.001)

    def forward(self, x):
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

class MaxPool3dTFPadding(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool3dTFPadding, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size, stride, padding)

    def compute_pad(self, dim_size, kernel_size, stride):
        if dim_size % stride == 0:
            pad_total = max(kernel_size - stride, 0)
        else:
            pad_total = max(kernel_size - (dim_size % stride), 0)
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return pad_before, pad_after

    def forward(self, x):
        pad_d = self.compute_pad(x.size(2), self.pool.kernel_size[0], self.pool.stride[0])
        pad_h = self.compute_pad(x.size(3), self.pool.kernel_size[1], self.pool.stride[1])
        pad_w = self.compute_pad(x.size(4), self.pool.kernel_size[2], self.pool.stride[2])

        x = F.pad(x, [pad_w[0], pad_w[1], pad_h[0], pad_h[1], pad_d[0], pad_d[1]])
        x = self.pool(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        assert len(out_channels) == 6

        self.b0 = Unit3D(in_channels, out_channels[0],
                         kernel_size=(1, 1, 1),
                         name=name + '/Branch_0/Conv3d_0a_1x1')

        self.b1a = Unit3D(in_channels, out_channels[1],
                          kernel_size=(1, 1, 1),
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(out_channels[1], out_channels[2],
                          kernel_size=(3, 3, 3),
                          name=name + '/Branch_1/Conv3d_0b_3x3')

        self.b2a = Unit3D(in_channels, out_channels[3],
                          kernel_size=(1, 1, 1),
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(out_channels[3], out_channels[4],
                          kernel_size=(3, 3, 3),
                          name=name + '/Branch_2/Conv3d_0b_3x3')

        self.b3a = MaxPool3dTFPadding(kernel_size=(3, 3, 3),
                                      stride=(1, 1, 1))
        self.b3b = Unit3D(in_channels, out_channels[5],
                          kernel_size=(1, 1, 1),
                          name=name + '/Branch_3/Conv3d_0b_1x1')

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1a(x)
        b1 = self.b1b(b1)
        b2 = self.b2a(x)
        b2 = self.b2b(b2)
        b3 = self.b3a(x)
        b3 = self.b3b(b3)
        return torch.cat([b0, b1, b2, b3], dim=1)

class InceptionI3d(nn.Module):
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', in_channels=3, name='inception_i3d'):  # ✨ thêm in_channels
        super(InceptionI3d, self).__init__()

        if final_endpoint not in [
            'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1',
            'Conv3d_2c_3x3', 'MaxPool3d_3a_3x3', 'Mixed_3b',
            'Mixed_3c', 'MaxPool3d_4a_3x3', 'Mixed_4b', 'Mixed_4c',
            'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 'MaxPool3d_5a_2x2',
            'Mixed_5b', 'Mixed_5c', 'Logits', 'Predictions']:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        self.end_points = {}
        self.final_endpoint = final_endpoint

        self._in_channels = in_channels  # ✨ lưu in_channels để truyền vào Conv3d đầu tiên

        self.end_points['Conv3d_1a_7x7'] = Unit3D(self._in_channels, 64,  # ✨ dùng self._in_channels
                                                  kernel_size=(7, 7, 7),
                                                  stride=(2, 2, 2),
                                                  name='Conv3d_1a_7x7')

        self.end_points['MaxPool3d_2a_3x3'] = MaxPool3dTFPadding(kernel_size=(1, 3, 3),
                                                                 stride=(1, 2, 2))
        self.end_points['Conv3d_2b_1x1'] = Unit3D(64, 64,
                                                  kernel_size=(1, 1, 1),
                                                  name='Conv3d_2b_1x1')
        self.end_points['Conv3d_2c_3x3'] = Unit3D(64, 192,
                                                  kernel_size=(3, 3, 3),
                                                  name='Conv3d_2c_3x3')
        self.end_points['MaxPool3d_3a_3x3'] = MaxPool3dTFPadding(kernel_size=(1, 3, 3),
                                                                 stride=(1, 2, 2))
        self.end_points['Mixed_3b'] = InceptionModule(192, [64, 96, 128, 16, 32, 32], 'Mixed_3b')
        self.end_points['Mixed_3c'] = InceptionModule(256, [128, 128, 192, 32, 96, 64], 'Mixed_3c')
        self.end_points['MaxPool3d_4a_3x3'] = MaxPool3dTFPadding(kernel_size=(3, 3, 3),
                                                                 stride=(2, 2, 2))
        self.end_points['Mixed_4b'] = InceptionModule(480, [192, 96, 208, 16, 48, 64], 'Mixed_4b')
        self.end_points['Mixed_4c'] = InceptionModule(512, [160, 112, 224, 24, 64, 64], 'Mixed_4c')
        self.end_points['Mixed_4d'] = InceptionModule(512, [128, 128, 256, 24, 64, 64], 'Mixed_4d')
        self.end_points['Mixed_4e'] = InceptionModule(512, [112, 144, 288, 32, 64, 64], 'Mixed_4e')
        self.end_points['Mixed_4f'] = InceptionModule(528, [256, 160, 320, 32, 128, 128], 'Mixed_4f')
        self.end_points['MaxPool3d_5a_2x2'] = MaxPool3dTFPadding(kernel_size=(2, 2, 2),
                                                                 stride=(2, 2, 2))
        self.end_points['Mixed_5b'] = InceptionModule(832, [256, 160, 320, 32, 128, 128], 'Mixed_5b')
        self.end_points['Mixed_5c'] = InceptionModule(832, [384, 192, 384, 48, 128, 128], 'Mixed_5c')

        self.build()

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.conv3d_0c_1x1 = Unit3D(1024, num_classes,
                                    kernel_size=(1, 1, 1),
                                    activation_fn=None,
                                    use_batch_norm=False,
                                    use_bias=True,
                                    name='Conv3d_0c_1x1')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def extract_features(self, x):
        for end_point in self.end_points:
            x = getattr(self, end_point)(x)
            if end_point == 'Mixed_5c':
                break
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.conv3d_0c_1x1(x)
        x = x.squeeze(3).squeeze(3).squeeze(2)
        return x
