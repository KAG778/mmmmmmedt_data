import torch
import torch.nn as nn
from torch.nn import functional as F

from models.GPT import GPT


################################################################################################################
# class MeDT
#
# We make embeddings of each input in the sequence. We then add position embeddings and feed embeddings
# to GPT transformer
# 
# Three different settings, 
#       BC: Behaviour Cloning
#       DT: Decision Transformer
#       MeDT: Medical Decision Transformer
#       
################################################################################################################
class MeDT(GPT):

    def _masked_mse_loss(self, pred, target, traj_mask=None):
        if traj_mask is None:
            return F.mse_loss(pred, target)
        token_loss = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        mask = traj_mask.type_as(token_loss)
        denom = mask.sum().clamp(min=1.0)
        return (token_loss * mask).sum() / denom

    # state, action, and return
    def forward(
        self,
        states,
        actions,
        targets=None,
        rtgs=None,
        timesteps=None,
        saps=None,
        divSaps=None,
        traj_len=None,
        task_embeddings=None,
        hindsight_embeddings=None,
        foresight_embeddings=None,
        traj_mask=None,
        is_visual=False,
        delta_saps2=None,
    ):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)
       
        state_embeddings = self.state_emb(states.type(torch.float32))
        token_device = state_embeddings.device
        

        if actions is not None and self.model_type == 'DT': 

            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        
            my_pos_emb = torch.zeros(
                timesteps.shape[0], timesteps.shape[1]*3, self.config.n_embd, device=token_device
            )
            my_pos_emb[:,0::3,:] = timesteps
            my_pos_emb[:,1::3,:] = timesteps
            my_pos_emb[:,2::3,:] = timesteps   

        elif actions is None and self.model_type == 'DT': # only happens at very first timestep of evaluation

            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]

            my_pos_emb = torch.zeros(
                timesteps.shape[0], timesteps.shape[1]*2, self.config.n_embd, device=token_device
            )
            my_pos_emb[:,0::2,:] = timesteps
            my_pos_emb[:,1::2,:] = timesteps
                    

        elif actions is not None and self.model_type == 'MeDT': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
            
            card_embeddings = self.card_emb(divSaps[:,:,0].unsqueeze(-1).type(torch.float32))
            resp_embeddings = self.resp_emb(divSaps[:,:,1].unsqueeze(-1).type(torch.float32))
            neur_embeddings = self.neur_emb(divSaps[:,:,2].unsqueeze(-1).type(torch.float32))
            ren_embeddings = self.ren_emb(divSaps[:,:,3].unsqueeze(-1).type(torch.float32))
            hep_embeddings = self.hep_emb(divSaps[:,:,4].unsqueeze(-1).type(torch.float32))
            haem_embeddings = self.haem_emb(divSaps[:,:,5].unsqueeze(-1).type(torch.float32))
            oth_embeddings = self.oth_emb(divSaps[:,:,6].unsqueeze(-1).type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*10 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::10,:] = rtg_embeddings
            token_embeddings[:,1::10,:] = card_embeddings
            token_embeddings[:,2::10,:] = resp_embeddings
            token_embeddings[:,3::10,:] = neur_embeddings
            token_embeddings[:,4::10,:] = ren_embeddings
            token_embeddings[:,5::10,:] = hep_embeddings
            token_embeddings[:,6::10,:] = haem_embeddings  
            token_embeddings[:,7::10,:] = oth_embeddings                       
            token_embeddings[:,8::10,:] = state_embeddings
            token_embeddings[:,9::10,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        
            my_pos_emb = torch.zeros(
                timesteps.shape[0],
                states.shape[1]*10 - int(targets is None),
                self.config.n_embd,
                device=token_device,
            )
            my_pos_emb[:,0::10,:] = timesteps
            my_pos_emb[:,1::10,:] = timesteps
            my_pos_emb[:,2::10,:] = timesteps                   
            my_pos_emb[:,3::10,:] = timesteps
            my_pos_emb[:,4::10,:] = timesteps
            my_pos_emb[:,5::10,:] = timesteps  
            my_pos_emb[:,6::10,:] = timesteps                   
            my_pos_emb[:,7::10,:] = timesteps
            my_pos_emb[:,8::10,:] = timesteps
            my_pos_emb[:,9::10,:] = timesteps[:,-states.shape[1] + int(targets is None):,:]


        elif actions is None and self.model_type == 'MeDT': # only happens at very first timestep of evaluation
          
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            card_embeddings = self.card_emb(divSaps[:,:,0].unsqueeze(-1).type(torch.float32))
            resp_embeddings = self.resp_emb(divSaps[:,:,1].unsqueeze(-1).type(torch.float32))
            neur_embeddings = self.neur_emb(divSaps[:,:,2].unsqueeze(-1).type(torch.float32))
            ren_embeddings = self.ren_emb(divSaps[:,:,3].unsqueeze(-1).type(torch.float32))
            hep_embeddings = self.hep_emb(divSaps[:,:,4].unsqueeze(-1).type(torch.float32))
            haem_embeddings = self.haem_emb(divSaps[:,:,5].unsqueeze(-1).type(torch.float32))
            oth_embeddings = self.oth_emb(divSaps[:,:,6].unsqueeze(-1).type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*9 , self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::9,:] = rtg_embeddings
            token_embeddings[:,1::9,:] = card_embeddings
            token_embeddings[:,2::9,:] = resp_embeddings
            token_embeddings[:,3::9,:] = neur_embeddings
            token_embeddings[:,4::9,:] = ren_embeddings
            token_embeddings[:,5::9,:] = hep_embeddings
            token_embeddings[:,6::9,:] = haem_embeddings  
            token_embeddings[:,7::9,:] = oth_embeddings
            token_embeddings[:,8::9,:] = state_embeddings

            my_pos_emb = torch.zeros(
                timesteps.shape[0], timesteps.shape[1]*9, self.config.n_embd, device=token_device
            )
            my_pos_emb[:,0::9,:] = timesteps
            my_pos_emb[:,1::9,:] = timesteps
            my_pos_emb[:,2::9,:] = timesteps                   
            my_pos_emb[:,3::9,:] = timesteps
            my_pos_emb[:,4::9,:] = timesteps
            my_pos_emb[:,5::9,:] = timesteps  
            my_pos_emb[:,6::9,:] = timesteps                   
            my_pos_emb[:,7::9,:] = timesteps
            my_pos_emb[:,8::9,:] = timesteps
	
        elif actions is not None and self.model_type == 'HFDT':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))

            if divSaps is None:
                foresight_signal = torch.zeros(
                    (states.shape[0], states.shape[1], 1),
                    dtype=torch.float32,
                    device=state_embeddings.device,
                )
            else:
                # Aggregate seven SAPS constituents as a compact acuity signal.
                foresight_signal = divSaps.type(torch.float32).sum(dim=-1, keepdim=True)

            # Previous-step acuity proxy as hindsight token.
            hindsight_signal = torch.cat(
                [foresight_signal[:, :1, :], foresight_signal[:, :-1, :]],
                dim=1,
            )
            hindsight_embeddings = self.saps_emb(hindsight_signal)
            foresight_embeddings = self.saps_emb(foresight_signal)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1]*5 - int(targets is None), self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::5, :] = rtg_embeddings
            token_embeddings[:, 1::5, :] = hindsight_embeddings
            token_embeddings[:, 2::5, :] = foresight_embeddings
            token_embeddings[:, 3::5, :] = state_embeddings
            token_embeddings[:, 4::5, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]

            my_pos_emb = torch.zeros(
                timesteps.shape[0],
                states.shape[1]*5 - int(targets is None),
                self.config.n_embd,
                device=token_device,
            )
            my_pos_emb[:, ::5, :] = timesteps
            my_pos_emb[:, 1::5, :] = timesteps
            my_pos_emb[:, 2::5, :] = timesteps
            my_pos_emb[:, 3::5, :] = timesteps
            my_pos_emb[:, 4::5, :] = timesteps[:, -states.shape[1] + int(targets is None):, :]

        elif actions is None and self.model_type == 'HFDT':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            if divSaps is None:
                foresight_signal = torch.zeros(
                    (states.shape[0], states.shape[1], 1),
                    dtype=torch.float32,
                    device=state_embeddings.device,
                )
            else:
                foresight_signal = divSaps.type(torch.float32).sum(dim=-1, keepdim=True)

            hindsight_signal = torch.cat(
                [foresight_signal[:, :1, :], foresight_signal[:, :-1, :]],
                dim=1,
            )
            hindsight_embeddings = self.saps_emb(hindsight_signal)
            foresight_embeddings = self.saps_emb(foresight_signal)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1]*4, self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::4, :] = rtg_embeddings
            token_embeddings[:, 1::4, :] = hindsight_embeddings
            token_embeddings[:, 2::4, :] = foresight_embeddings
            token_embeddings[:, 3::4, :] = state_embeddings

            my_pos_emb = torch.zeros(
                timesteps.shape[0],
                timesteps.shape[1]*4,
                self.config.n_embd,
                device=token_device,
            )
            my_pos_emb[:, ::4, :] = timesteps
            my_pos_emb[:, 1::4, :] = timesteps
            my_pos_emb[:, 2::4, :] = timesteps
            my_pos_emb[:, 3::4, :] = timesteps

        elif actions is not None and self.model_type == 'SeMDT':
            if task_embeddings is None or hindsight_embeddings is None or foresight_embeddings is None:
                raise ValueError("SeMDT requires task/hindsight/foresight embeddings.")

            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))
            task_token_embeddings = self.task_text_emb(task_embeddings.type(torch.float32))
            hindsight_token_embeddings = self.h_text_emb(hindsight_embeddings.type(torch.float32))
            foresight_token_embeddings = self.f_text_emb(foresight_embeddings.type(torch.float32))

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1]*6 - int(targets is None), self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::6, :] = task_token_embeddings
            token_embeddings[:, 1::6, :] = rtg_embeddings
            token_embeddings[:, 2::6, :] = hindsight_token_embeddings
            token_embeddings[:, 3::6, :] = foresight_token_embeddings
            token_embeddings[:, 4::6, :] = state_embeddings
            token_embeddings[:, 5::6, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]

            my_pos_emb = torch.zeros(
                timesteps.shape[0],
                states.shape[1]*6 - int(targets is None),
                self.config.n_embd,
                device=token_device,
            )
            my_pos_emb[:, ::6, :] = timesteps
            my_pos_emb[:, 1::6, :] = timesteps
            my_pos_emb[:, 2::6, :] = timesteps
            my_pos_emb[:, 3::6, :] = timesteps
            my_pos_emb[:, 4::6, :] = timesteps
            my_pos_emb[:, 5::6, :] = timesteps[:, -states.shape[1] + int(targets is None):, :]

        elif actions is None and self.model_type == 'SeMDT':
            if task_embeddings is None or hindsight_embeddings is None or foresight_embeddings is None:
                raise ValueError("SeMDT requires task/hindsight/foresight embeddings.")

            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            task_token_embeddings = self.task_text_emb(task_embeddings.type(torch.float32))
            hindsight_token_embeddings = self.h_text_emb(hindsight_embeddings.type(torch.float32))
            foresight_token_embeddings = self.f_text_emb(foresight_embeddings.type(torch.float32))

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1]*5, self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::5, :] = task_token_embeddings
            token_embeddings[:, 1::5, :] = rtg_embeddings
            token_embeddings[:, 2::5, :] = hindsight_token_embeddings
            token_embeddings[:, 3::5, :] = foresight_token_embeddings
            token_embeddings[:, 4::5, :] = state_embeddings

            my_pos_emb = torch.zeros(
                timesteps.shape[0],
                timesteps.shape[1]*5,
                self.config.n_embd,
                device=token_device,
            )
            my_pos_emb[:, ::5, :] = timesteps
            my_pos_emb[:, 1::5, :] = timesteps
            my_pos_emb[:, 2::5, :] = timesteps
            my_pos_emb[:, 3::5, :] = timesteps
            my_pos_emb[:, 4::5, :] = timesteps

        # ── 7-token variants with ATG (delta_saps2) ──
        # SeMDT_ATG_A: [task, ATG, RTG, hindsight, foresight, state, action]
        elif actions is not None and self.model_type == 'SeMDT_ATG_A':
            if task_embeddings is None or hindsight_embeddings is None or foresight_embeddings is None:
                raise ValueError("SeMDT_ATG_A requires task/hindsight/foresight embeddings.")
            if delta_saps2 is None:
                raise ValueError("SeMDT_ATG_A requires delta_saps2.")

            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            atg_embeddings = self.atg_emb(delta_saps2.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))
            task_token_embeddings = self.task_text_emb(task_embeddings.type(torch.float32))
            hindsight_token_embeddings = self.h_text_emb(hindsight_embeddings.type(torch.float32))
            foresight_token_embeddings = self.f_text_emb(foresight_embeddings.type(torch.float32))

            T = states.shape[1]
            seq_len = T * 7 - int(targets is None)
            token_embeddings = torch.zeros(
                (states.shape[0], seq_len, self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::7, :] = task_token_embeddings
            token_embeddings[:, 1::7, :] = atg_embeddings
            token_embeddings[:, 2::7, :] = rtg_embeddings
            token_embeddings[:, 3::7, :] = hindsight_token_embeddings
            token_embeddings[:, 4::7, :] = foresight_token_embeddings
            token_embeddings[:, 5::7, :] = state_embeddings
            token_embeddings[:, 6::7, :] = action_embeddings[:, -T + int(targets is None):, :]

            my_pos_emb = torch.zeros(
                timesteps.shape[0], seq_len, self.config.n_embd, device=token_device,
            )
            my_pos_emb[:, ::7, :] = timesteps
            my_pos_emb[:, 1::7, :] = timesteps
            my_pos_emb[:, 2::7, :] = timesteps
            my_pos_emb[:, 3::7, :] = timesteps
            my_pos_emb[:, 4::7, :] = timesteps
            my_pos_emb[:, 5::7, :] = timesteps
            my_pos_emb[:, 6::7, :] = timesteps[:, -T + int(targets is None):, :]

        elif actions is None and self.model_type == 'SeMDT_ATG_A':
            if task_embeddings is None or hindsight_embeddings is None or foresight_embeddings is None:
                raise ValueError("SeMDT_ATG_A requires task/hindsight/foresight embeddings.")
            if delta_saps2 is None:
                raise ValueError("SeMDT_ATG_A requires delta_saps2.")

            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            atg_embeddings = self.atg_emb(delta_saps2.type(torch.float32))
            task_token_embeddings = self.task_text_emb(task_embeddings.type(torch.float32))
            hindsight_token_embeddings = self.h_text_emb(hindsight_embeddings.type(torch.float32))
            foresight_token_embeddings = self.f_text_emb(foresight_embeddings.type(torch.float32))

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1]*6, self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::6, :] = task_token_embeddings
            token_embeddings[:, 1::6, :] = atg_embeddings
            token_embeddings[:, 2::6, :] = rtg_embeddings
            token_embeddings[:, 3::6, :] = hindsight_token_embeddings
            token_embeddings[:, 4::6, :] = foresight_token_embeddings
            token_embeddings[:, 5::6, :] = state_embeddings

            my_pos_emb = torch.zeros(
                timesteps.shape[0], states.shape[1]*6, self.config.n_embd, device=token_device,
            )
            my_pos_emb[:, ::6, :] = timesteps
            my_pos_emb[:, 1::6, :] = timesteps
            my_pos_emb[:, 2::6, :] = timesteps
            my_pos_emb[:, 3::6, :] = timesteps
            my_pos_emb[:, 4::6, :] = timesteps
            my_pos_emb[:, 5::6, :] = timesteps

        # SeMDT_ATG_B: [task, RTG, ATG, hindsight, foresight, state, action]
        elif actions is not None and self.model_type == 'SeMDT_ATG_B':
            if task_embeddings is None or hindsight_embeddings is None or foresight_embeddings is None:
                raise ValueError("SeMDT_ATG_B requires task/hindsight/foresight embeddings.")
            if delta_saps2 is None:
                raise ValueError("SeMDT_ATG_B requires delta_saps2.")

            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            atg_embeddings = self.atg_emb(delta_saps2.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))
            task_token_embeddings = self.task_text_emb(task_embeddings.type(torch.float32))
            hindsight_token_embeddings = self.h_text_emb(hindsight_embeddings.type(torch.float32))
            foresight_token_embeddings = self.f_text_emb(foresight_embeddings.type(torch.float32))

            T = states.shape[1]
            seq_len = T * 7 - int(targets is None)
            token_embeddings = torch.zeros(
                (states.shape[0], seq_len, self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::7, :] = task_token_embeddings
            token_embeddings[:, 1::7, :] = rtg_embeddings
            token_embeddings[:, 2::7, :] = atg_embeddings
            token_embeddings[:, 3::7, :] = hindsight_token_embeddings
            token_embeddings[:, 4::7, :] = foresight_token_embeddings
            token_embeddings[:, 5::7, :] = state_embeddings
            token_embeddings[:, 6::7, :] = action_embeddings[:, -T + int(targets is None):, :]

            my_pos_emb = torch.zeros(
                timesteps.shape[0], seq_len, self.config.n_embd, device=token_device,
            )
            my_pos_emb[:, ::7, :] = timesteps
            my_pos_emb[:, 1::7, :] = timesteps
            my_pos_emb[:, 2::7, :] = timesteps
            my_pos_emb[:, 3::7, :] = timesteps
            my_pos_emb[:, 4::7, :] = timesteps
            my_pos_emb[:, 5::7, :] = timesteps
            my_pos_emb[:, 6::7, :] = timesteps[:, -T + int(targets is None):, :]

        elif actions is None and self.model_type == 'SeMDT_ATG_B':
            if task_embeddings is None or hindsight_embeddings is None or foresight_embeddings is None:
                raise ValueError("SeMDT_ATG_B requires task/hindsight/foresight embeddings.")
            if delta_saps2 is None:
                raise ValueError("SeMDT_ATG_B requires delta_saps2.")

            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            atg_embeddings = self.atg_emb(delta_saps2.type(torch.float32))
            task_token_embeddings = self.task_text_emb(task_embeddings.type(torch.float32))
            hindsight_token_embeddings = self.h_text_emb(hindsight_embeddings.type(torch.float32))
            foresight_token_embeddings = self.f_text_emb(foresight_embeddings.type(torch.float32))

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1]*6, self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::6, :] = task_token_embeddings
            token_embeddings[:, 1::6, :] = rtg_embeddings
            token_embeddings[:, 2::6, :] = atg_embeddings
            token_embeddings[:, 3::6, :] = hindsight_token_embeddings
            token_embeddings[:, 4::6, :] = foresight_token_embeddings
            token_embeddings[:, 5::6, :] = state_embeddings

            my_pos_emb = torch.zeros(
                timesteps.shape[0], states.shape[1]*6, self.config.n_embd, device=token_device,
            )
            my_pos_emb[:, ::6, :] = timesteps
            my_pos_emb[:, 1::6, :] = timesteps
            my_pos_emb[:, 2::6, :] = timesteps
            my_pos_emb[:, 3::6, :] = timesteps
            my_pos_emb[:, 4::6, :] = timesteps
            my_pos_emb[:, 5::6, :] = timesteps

        elif actions is not None and self.model_type == 'BC':
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
       
            my_pos_emb = torch.zeros(
                timesteps.shape[0], timesteps.shape[1]*2, self.config.n_embd, device=token_device
            )
            my_pos_emb[:,0::2,:] = timesteps
            my_pos_emb[:,1::2,:] = timesteps
       
        elif actions is None and self.model_type == 'BC': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings

            my_pos_emb = torch.zeros(
                token_embeddings.shape[0],
                token_embeddings.shape[1],
                token_embeddings.shape[2],
                device=token_device,
            )
            my_pos_emb[:,:,:] = timesteps


        else:
            raise NotImplementedError()

        # Add position embeddings
        position_embeddings =  self.pos_emb[:, :token_embeddings.shape[1], :] + my_pos_emb[:, :token_embeddings.shape[1], :]
        x = self.drop(token_embeddings) + position_embeddings
            
        for idx, block in enumerate(self.blocks):
            x, attn_score = block(x)
            if is_visual:
                from visualization import visualize_attention
                visualize_attention(attn_score, idx)
                self.attn_score = attn_score

        x = self.ln_f(x)
        logits = self.head(x)
  
        action_loss = None

        if actions is not None and self.model_type == 'DT':
            logits = logits[:, 1::3, :] 
        elif actions is None and self.model_type == 'DT':
            logits = logits[:, 1::2, :]            
        elif actions is not None and self.model_type == 'MeDT':
            logits = logits[:, 8::10, :]
        elif actions is None and self.model_type == 'MeDT':
            logits = logits[:, 8::9, :]            
        elif actions is not None and self.model_type == 'HFDT':
            logits = logits[:, 3::5, :]
        elif actions is None and self.model_type == 'HFDT':
            logits = logits[:, 3::4, :]
        elif actions is not None and self.model_type == 'SeMDT':
            logits = logits[:, 4::6, :]
        elif actions is None and self.model_type == 'SeMDT':
            logits = logits[:, 4::5, :]
        elif actions is not None and self.model_type == 'SeMDT_ATG_A':
            logits = logits[:, 5::7, :]
        elif actions is None and self.model_type == 'SeMDT_ATG_A':
            logits = logits[:, 5::6, :]
        elif actions is not None and self.model_type == 'SeMDT_ATG_B':
            logits = logits[:, 5::7, :]
        elif actions is None and self.model_type == 'SeMDT_ATG_B':
            logits = logits[:, 5::6, :]
        elif actions is not None and self.model_type == 'BC':
            logits = logits[:, ::2, :] 
        elif actions is None and self.model_type == 'BC':
            logits = logits
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss        
        if targets is not None:
            if traj_mask is None:
                action_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            else:
                per_token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape_as(targets)
                mask = traj_mask.reshape_as(targets).type_as(per_token_loss)
                denom = mask.sum().clamp(min=1.0)
                action_loss = (per_token_loss * mask).sum() / denom
                    
        return logits, action_loss, self.attn_score

    def forward_joint(
        self,
        states,
        actions,
        targets=None,
        rtgs=None,
        timesteps=None,
        saps=None,
        divSaps=None,
        traj_len=None,
        task_embeddings=None,
        hindsight_embeddings=None,
        foresight_embeddings=None,
        traj_mask=None,
        state_targets=None,
        state_loss_weight=1.0,
        action_loss_weight=1.0,
    ):
        if self.model_type != "SeMDT":
            raise NotImplementedError("forward_joint is currently implemented for SeMDT only.")
        if actions is None:
            raise ValueError("forward_joint requires action tokens to predict next-state targets.")
        if task_embeddings is None or hindsight_embeddings is None or foresight_embeddings is None:
            raise ValueError("SeMDT joint training requires task/hindsight/foresight embeddings.")

        state_embeddings = self.state_emb(states.type(torch.float32))
        token_device = state_embeddings.device
        rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
        action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))
        task_token_embeddings = self.task_text_emb(task_embeddings.type(torch.float32))
        hindsight_token_embeddings = self.h_text_emb(hindsight_embeddings.type(torch.float32))
        foresight_token_embeddings = self.f_text_emb(foresight_embeddings.type(torch.float32))

        token_embeddings = torch.zeros(
            (states.shape[0], states.shape[1] * 6, self.config.n_embd),
            dtype=torch.float32,
            device=state_embeddings.device,
        )
        token_embeddings[:, ::6, :] = task_token_embeddings
        token_embeddings[:, 1::6, :] = rtg_embeddings
        token_embeddings[:, 2::6, :] = hindsight_token_embeddings
        token_embeddings[:, 3::6, :] = foresight_token_embeddings
        token_embeddings[:, 4::6, :] = state_embeddings
        token_embeddings[:, 5::6, :] = action_embeddings

        my_pos_emb = torch.zeros(
            timesteps.shape[0],
            states.shape[1] * 6,
            self.config.n_embd,
            device=token_device,
        )
        my_pos_emb[:, ::6, :] = timesteps
        my_pos_emb[:, 1::6, :] = timesteps
        my_pos_emb[:, 2::6, :] = timesteps
        my_pos_emb[:, 3::6, :] = timesteps
        my_pos_emb[:, 4::6, :] = timesteps
        my_pos_emb[:, 5::6, :] = timesteps

        position_embeddings = self.pos_emb[:, : token_embeddings.shape[1], :] + my_pos_emb
        x = self.drop(token_embeddings) + position_embeddings

        for block in self.blocks:
            x, _attn_score = block(x)

        x = self.ln_f(x)
        action_hidden = x[:, 4::6, :]
        action_logits = self.head(action_hidden)

        # Predict next real patient state from the action-token hidden representation.
        next_state_hidden = x[:, 5::6, :]
        state_preds = self.state_head(next_state_hidden)

        action_loss = None
        state_loss = None
        total_loss = None

        if targets is not None:
            if traj_mask is None:
                action_loss = F.cross_entropy(
                    action_logits.reshape(-1, action_logits.size(-1)),
                    targets.reshape(-1),
                )
            else:
                per_token_loss = F.cross_entropy(
                    action_logits.reshape(-1, action_logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape_as(targets)
                mask = traj_mask.reshape_as(targets).type_as(per_token_loss)
                denom = mask.sum().clamp(min=1.0)
                action_loss = (per_token_loss * mask).sum() / denom

        if state_targets is not None:
            pred = state_preds[:, :-1, :]
            target = state_targets[:, 1:, :]
            state_mask = None if traj_mask is None else traj_mask[:, 1:, 0]
            state_loss = self._masked_mse_loss(pred, target, state_mask)

        if action_loss is not None or state_loss is not None:
            total_loss = 0.0
            if action_loss is not None:
                total_loss = total_loss + action_loss_weight * action_loss
            if state_loss is not None:
                total_loss = total_loss + state_loss_weight * state_loss

        return {
            "action_logits": action_logits,
            "state_preds": state_preds,
            "action_loss": action_loss,
            "state_loss": state_loss,
            "total_loss": total_loss,
        }
