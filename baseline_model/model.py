import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class RNBlock(nn.Module):
    def __init__(self, n_in, n_hid, n_out, residual=False):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_out)
        self.residual = residual

    @staticmethod
    def edge2node(x, rel_rec):
        incoming = torch.matmul(torch.transpose(rel_rec, 1, 2), x)
        return incoming / incoming.size(1)

    @staticmethod
    def node2edge(x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], -1)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        '''inputs: B x N x D
           rel_rec: B x M x N
           rel_send: B x M x N
        '''
        x = self.node2edge(inputs, rel_rec, rel_send)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        x = self.edge2node(x, rel_rec)
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        if self.residual:
            x = x + inputs
        return x


class GATBlock(nn.Module):
    def __init__(self, n_in, n_hid, n_head):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid*2, n_head)

    @staticmethod
    def edge2node(x, a, rel_rec):
        feats = []
        # print(x.shape, a.shape, rel_rec.shape)
        # print(rel_rec.sum(1))
        for i in range(rel_rec.size(0)):
            feats_ins = []
            for j in range(rel_rec.size(2)):
                feat = x[i, rel_rec[i, :, j]==1, :]
                att = F.softmax(a[i, rel_rec[i, :, j]==1, :], dim=0)
                feat = att.unsqueeze(-1) @ feat.unsqueeze(-2)
                feat = feat.sum(dim=0).view(-1)
                feats_ins.append(feat)
            feats.append(torch.stack(feats_ins, 0))
        return torch.stack(feats, dim=0)

    @staticmethod
    def node2edge(x, rel_send, rel_rec=None):
        senders = torch.matmul(rel_send, x)
        if rel_rec is not None:
            receivers = torch.matmul(rel_rec, x)
            return torch.cat([senders, receivers], -1)
        else:
            return senders

    def forward(self, inputs, rel_rec, rel_send):
        '''inputs: B x N x D
           rel_rec: B x M x N
           rel_send: B x M x N
        '''
        x = self.fc1(inputs)
        a = F.elu(self.fc2(self.node2edge(x, rel_send, rel_rec)))
        x = F.elu(self.edge2node(self.node2edge(x, rel_send), a, rel_rec))
        return x


class GRU(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        d = 2
        h = 128
        self.fc1 = nn.Linear(d, h)
        self.gru = nn.GRUCell(h, h)
        self.out = nn.Linear(h, d)

    def forward(self, inputs, hidden, mask, label=None):
        '''inputs: T x B x N x D
           hidden: B x N x H
           mask: B x N (numpy array)
           label: B x N x D
        '''
        B, N, H = hidden.size()
        h = hidden
        for i in range(inputs.size(0)):
            x = self.fc1(inputs[i])
            h = self.gru(x.view(-1, H), h.view(-1, H)).view(B, N, H)
        pred = self.out(h)
        if label is None:
            return pred
        else:
            mask = mask.float().cuda()
            dist = torch.sum((pred - label) ** 2, 2) * mask
            loss = torch.mean(dist.sum(1) / mask.sum(1))
            return pred, loss


class RRNN(nn.Module):
    def __init__(self, d=2, h=128 ):
        nn.Module.__init__(self)
        # self.fc1 = nn.Linear(d, h)
        self.rn1 = RNBlock(d*2, h, h)
        self.rn2 = RNBlock(h*2, h, h, residual=True)
        self.gru = nn.GRUCell(h, h)
        self.out = nn.Linear(h, d)

    @staticmethod
    def get_graph(nodes, n):
        receivers = np.zeros([n*(n-1), n], np.float32)
        senders = np.zeros([n*(n-1), n], np.float32)
        k = 0
        for i in nodes:
            for j in nodes:
                senders[k, i] = 1
                receivers[k, j] = 1
                k += 1
        return receivers, senders

    def forward(self, inputs, hidden, mask, label=None):
        """
        Args:
            inputs: T x B x N (17 joints) x D (3d position)
            hidden: B x N x H
            mask: B x N (numpy array)
            label: B x N x D
        """

        B, N, H = hidden.size()
        # rel_rec, rel_send = zip(*[self.get_graph(np.where(mask.cpu()[i])[0], mask.cpu().shape[1])
        #     for i in range(mask.cpu().shape[0])])        
        
        rel_rec, rel_send = zip(*[self.get_graph( np.where(mask.cpu()[i])[0], mask.cpu().shape[1]) for i in range(mask.cpu().shape[0]) ])
        
        rel_rec = torch.from_numpy(np.stack(rel_rec)).cuda()
        rel_send = torch.from_numpy(np.stack(rel_send)).cuda()
        
        # print(rel_rec.shape, rel_send.shape)
        
        h = hidden
        for i in range(inputs.size(0)):
            # x = self.fc1(inputs[i])
            x = self.rn1(inputs[i], rel_rec, rel_send)
            h = self.gru(x.view(-1, H), h.view(-1, H)).view(B, N, H)
            h = self.rn2(h, rel_rec, rel_send)
        pred = self.out(h)
        if label is None:
            return pred
        else:
            mask = mask.float().cuda()
            dist = torch.sum((pred - label) ** 2, 2) * mask
            loss = torch.mean(dist.sum(1) / mask.sum(1))
            return pred, loss


class GATRNN(RRNN):
    def __init__(self):
        super(GATRNN, self).__init__()
        a = 4
        self.rn1 = GATBlock(d, h // a, a)
        self.rn2 = GATBlock(h, h // a, a)


class RRNN_var(RRNN):
    def __init__(self):
        super(RRNN_var, self).__init__()
        self.out = nn.Linear(128, 5)

    def forward(self, inputs, hidden, mask, label=None):
        '''inputs: T x B x N x D
           hidden: B x N x H
           mask: B x N (numpy array)
           label: B x N x D
        '''
        B, N, H = hidden.size()
        rel_rec, rel_send = zip(*[self.get_graph(np.where(mask.cpu()[i])[0], mask.cpu().shape[1])
            for i in range(mask.cpu().shape[0])])
        rel_rec = torch.from_numpy(np.stack(rel_rec)).cuda()
        rel_send = torch.from_numpy(np.stack(rel_send)).cuda()
        h = hidden
        for i in range(inputs.size(0)):
            x = self.rn1(inputs[i], rel_rec, rel_send)
            h = self.gru(x.view(-1, H), h.view(-1, H)).view(B, N, H)
            h = self.rn2(h, rel_rec, rel_send)
        pred = self.out(h).view(B*N, -1)
        mu = pred[:, :2]
        sigma = [[[p[2], p[4]], [p[4], p[3]]] for p in pred]
        sigma = torch.Tensor(sigma).cuda()
        dists = [MultivariateNormal(u, s) for u, s in zip(mu, sigma)]
        if label is None:
            sample = torch.stack([d.rsample() for d in dists]).view(B, N, -1)
            return sample
        else:
            mask.float().cuda()
            label = label.contiguous().view(B*N, -1)
            nll = torch.stack([-d.log_prob(label[i]) for i, d in enumerate(dists)]).view(B, N)
            loss = torch.mean((nll * mask).sum(1) / mask.sum(1))
            return pred, loss


if __name__ == "__main__":
    T = 8
    B = 16 # batch
    N = 50 # Sample per batch
    D = 2  # x, y, position
    H = 128# Hidden depth

    ## Test 
    T = 8
    B = 16 # batch
    N = 17 # 17 joints per sample
    D = 3  # x, y, z position
    H = 128# Hidden depth

    inputs = torch.rand(T, B, N, D).cuda()
    hidden = torch.zeros(B, N, H).cuda()
    mask = np.random.randint(2, size=(B, N))
    mask = torch.from_numpy(mask).float().cuda()
    label = torch.rand(B, N, D).cuda()
    
    model = RRNN(d=D, h=H)
    model.cuda()
    start = time.time()
    pred, loss = model(inputs, hidden, mask, label)
    print(time.time() - start)
    loss.backward()
    print(pred.shape)
    print(loss.item())

    embed_dim = 128
    num_heads = 8
    batch_size = 1
    seq_length = 10
    multihead_attn = nn.MultiheadAttention(embed_dim , num_heads)

    query = torch.randn([batch_size, seq_length, embed_dim])
    key = query
    value = query

    print ('Input size: ' + str(query.size()))

    attn_output, attn_output_weights = multihead_attn(query, key, value)

    print ('Output size: ' + str(attn_output.size()) + str(attn_output_weights.size()))