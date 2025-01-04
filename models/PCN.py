import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

def kern(period, k):
    while period > k:
        divisor = 2
        found_divisor = False

        # Try to find a divisor starting from 2 upwards
        while divisor <= period:
            if period % divisor == 0:
                period = period // divisor  # Use integer division
                found_divisor = True
                break
            divisor += 1

        # If no divisor found (which should not happen), return 1
        if not found_divisor:
            return 1

    # Return the final result
    if period < k:
        return period
    elif period == k:
        return 0


# back -linear   ;
# Normalization  ;
class Fconv(nn.Module):
    def __init__(self, configs):
        super(Fconv, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.max_k = configs.max_k
        self.d_model = configs.d_model
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(1, configs.d_model, (self.max_k, 1)))
        for i in range(1, self.max_k + 1):
            self.conv.append(
                nn.Conv2d(1, configs.d_model, (i, 1))
            )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            pe = kern(period, self.max_k)
            pad = pe - 1 if pe != 0 else self.max_k - 1
            out = torch.cat([x, x[:, -pad - 1:-1, :]], dim=1)
            out = out.unsqueeze(1)
            out = self.conv[pe](out)
            # reshape back
            out = out.reshape(B, -1, N)
            res.append(out[:, :, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, self.d_model * T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.model = Fconv(configs)
        self.seq2pred = nn.Linear(configs.seq_len * configs.d_model, configs.pred_len)
        if configs.relu == 'GELU':
            self.relu = nn.GELU()
        else:
            self.relu = None
    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_out = x_enc
        enc_out = self.model(enc_out)
        if self.relu is not None:
            enc_out = self.relu(enc_out)
        dec_out = self.seq2pred(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
