
import os

import torch
import lightning.pytorch as pl

import blocks


class AttentiveDeepSym(pl.LightningModule):

    def __init__(self, config):

        super(AttentiveDeepSym, self).__init__()
        self.save_hyperparameters()
        self._initialize_networks(config)
        self.lr = config["lr"]
        self.loss_coeff = config["loss_coeff"]

    def _initialize_networks(self, config):

        enc_layers = [config["state_dim"]] + \
                     [config["hidden_dim"]] * config["n_hidden_layers"] + \
                     [config["latent_dim"]]
        self.encoder = torch.nn.Sequential(
            blocks.MLP(enc_layers, last_layer_norm=True),
            blocks.GumbelSigmoidLayer(T=config["gumbel_t"])
        )
        

        pre_att_layers = [config["state_dim"]] + \
                         [config["hidden_dim"]] * config["n_hidden_layers"]
        self.pre_attention = blocks.MLP(pre_att_layers)
        

        self.attention = blocks.GumbelAttention(
            in_dim=config["hidden_dim"],
            out_dim=config["hidden_dim"],
            num_heads=config["n_attention_heads"],
            temperature=config["gumbel_t"]
        )


        post_enc_layers = [config["latent_dim"] + config["action_dim"]] + \
                          [config["hidden_dim"]] * config["n_hidden_layers"]
        self.post_encoder = blocks.MLP(post_enc_layers)

        dec_layers = [config["hidden_dim"] * config["n_attention_heads"]] + \
                     [config["hidden_dim"]] * config["n_hidden_layers"] + \
                     [config["effect_dim"]]
        self.decoder = blocks.MLP(dec_layers)

    def encode(self, x, eval_mode=False):

        n_sample, n_seg, n_feat = x.shape
        x = x.reshape(-1, n_feat)
        h = self.encoder(x)
        h = h.reshape(n_sample, n_seg, -1)
        if eval_mode:
            h = h.round()
        return h

    def concat(self, s, a, eval_mode=False):

        h = self.encode(s, eval_mode)
        z = torch.cat([h, a], dim=-1)
        return z

    def attn_weights(self, x, pad_mask, eval_mode=False):

        n_sample, n_seg, n_feat = x.shape
        x = x.reshape(-1, n_feat)
        x = self.pre_attention(x)
        x = x.reshape(n_sample, n_seg, -1)
        pad_mask = pad_mask.to(x.device)
        attn_weights = self.attention(x, src_key_mask=pad_mask)
        if eval_mode:
            attn_weights = attn_weights.round()
        return attn_weights

    def aggregate(self, z, attn_weights):

        n_batch, n_seg, n_dim = z.shape
        post_h = self.post_encoder(z.reshape(-1, n_dim)).reshape(n_batch, n_seg, -1).unsqueeze(1)
        att_out = attn_weights @ post_h
        att_out = att_out.permute(0, 2, 1, 3).reshape(n_batch, n_seg, -1)
        return att_out

    def decode(self, z, pad_mask):

        n_sample, n_seg, z_dim = z.shape
        z = z.reshape(-1, z_dim)
        e = self.decoder(z)
        e = e.reshape(n_sample, n_seg, -1)
        pad_mask = pad_mask.reshape(n_sample, n_seg, 1)
        e_masked = e * pad_mask
        return e_masked

    def forward(self, s, a, pad_mask, eval_mode=False):

        z = self.concat(s, a, eval_mode)
        attn_weights = self.attn_weights(s, pad_mask, eval_mode)
        z_att = self.aggregate(z, attn_weights)
        e = self.decode(z_att, pad_mask)
        return z, attn_weights, e

    def loss(self, e_pred, e, pad_mask):
    
        loss = torch.nn.functional.mse_loss(e_pred, e, reduction="none")
        loss = (loss * pad_mask.unsqueeze(2)).sum(dim=[1, 2]).mean() * self.loss_coeff
        return loss

    def training_step(self, batch, _):
        s, a, e, pad_mask, _ = batch
        _, _, e_pred = self.forward(s, a, pad_mask)
        loss = self.loss(e_pred, e, pad_mask)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        s, a, e, pad_mask, _ = batch
        _, _, e_pred = self.forward(s, a, pad_mask)
        loss = self.loss(e_pred, e, pad_mask)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        s, a, e, pad_mask, _ = batch
        _, _, e_pred = self.forward(s, a, pad_mask)
        return (e - e_pred).abs()

    def predict_step(self, batch, _):
        s, a, _, pad_mask, sn = batch
        z = self.encode(s, eval_mode=False)
        z = z * pad_mask.unsqueeze(2)
        r = self.attn_weights(s, pad_mask, eval_mode=False)
        zn = self.encode(sn, eval_mode=False)
        zn = zn * pad_mask.unsqueeze(2)
        rn = self.attn_weights(sn, pad_mask, eval_mode=False)
        return {"z": z, "r": r, "a": a, "zn": zn, "rn": rn, "m": pad_mask}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        norms = pl.utilities.grad_norm(self, norm_type=2)
        self.log_dict(norms)


def load_ckpt(name, tag="best"):

    save_dir = os.path.join("../logs", name)
    if os.path.exists(save_dir):
        ckpts = list(filter(lambda x: x.endswith(".ckpt"), os.listdir(save_dir)))
        if tag == "best":
            for ckpt in ckpts:
                if "last" not in ckpt:
                    break
        else:
            ckpt = "last.ckpt"
        ckpt_path = os.path.join(save_dir, ckpt)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = AttentiveDeepSym.load_from_checkpoint(ckpt_path, map_location=device)
    else:
        raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")
    return model, ckpt_path
