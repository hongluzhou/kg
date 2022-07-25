import os
import pdb

import torch
# import torch.nn.functional as F


class StepRegressionNCELoss(torch.nn.Module):
    def __init__(self, args, logger):
        super(StepRegressionNCELoss, self).__init__()
        
        self.args = args
        self.logger = logger
        
        from datasets.build_kg_no_edges import get_step_des_feats
        if args.adapter_pseudo_label_form == 'step_narraion_matching_mpnet':
            self.step_embed = get_step_des_feats(args, logger, language_model="MPNet")
        elif args.adapter_pseudo_label_form == 'step_video_matching_s3d_text':
            self.step_embed = get_step_des_feats(args, logger, language_model="S3D")
        else:
            logger.info('The adapter_pseudo_label_form is not implemented!\nFunc: {}\nFile:{}'.format(
                    __name__, __file__))
            os._exit(0)
        # self.step_embd: (S, d)
        self.step_embed = torch.tensor(self.step_embed).to(args.device)
        

    def forward(self, video_embed, video_labels):
        """
        - video_embed: (B, d)
        - video_labels: (B,)
        """
        x = torch.matmul(video_embed,  self.step_embed[video_labels].t())  # (B, B)
        x = x.view(video_embed.shape[0], video_embed.shape[0], -1)  # (B, B, 1)
        nominator = x * torch.eye(x.shape[0])[:,:,None].to(self.args.device)  # (B, B, 1)
        nominator = nominator.sum(dim=1)  # (B, 1)
        nominator = torch.logsumexp(nominator, dim=1)  # (B,)
        
        x = torch.matmul(video_embed, self.step_embed.t())  # (B, S)
        idx = torch.ones(video_embed.shape[0], self.step_embed.shape[0])  # (B, S)
        idx[torch.arange(video_embed.shape[0]), video_labels] = 0  # (B, S)
        denominator = x * idx.to(self.args.device)  # (B, S)
        denominator = torch.logsumexp(denominator, dim=1)  # (B,)
        
        return torch.mean(denominator - nominator), x
