import torch
import pdb


class MILNCELoss(torch.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = torch.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * torch.eye(x.shape[0])[:,:,None]#.cuda()
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(denominator - nominator)

    
# batch_size 1
# input raw video feat: torch.Size([1, 3, 16, 224, 224])
# input raw text feat (num_cannidiate=4): torch.Size([4, 20])
# model learned video feat: video_embd torch.Size([1, 512])
# modellearned text feat: torch.Size([1, 512])

batch_size = 10
hidden_dim = 4
video_embd = torch.rand(batch_size, hidden_dim)
text_embd = torch.rand(batch_size, hidden_dim)
    
loss_fun = MILNCELoss()
loss = loss_fun(video_embd, text_embd)
print('loss: {}'.format(loss))
