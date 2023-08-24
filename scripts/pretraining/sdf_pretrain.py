import numpy as np

import torch
import torch.nn.functional as F

from pathlib import Path

from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.sdf_field import SDFFieldConfig
from scripts.pretraining.sdf_mesh_extraction import extract_mesh

class SDFDataset(torch.utils.data.Dataset):

    def __init__(self, sdf_path, batch_size):
        self.path = sdf_path
        self.batch_size = batch_size
        self.combined = np.load(self.path, allow_pickle=True).item()
        self.positions = self.combined['samples'].astype(np.float32)
        self.sdf_values = self.combined['sdf'].astype(np.float32)
        if 'normals' in self.combined.keys():
            self.normals = self.combined['normals'].astype(np.float32)

    def __len__(self):
        # (Num samples) / (batch size)
        return int(self.positions.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        if idx >= int(self.positions.shape[0] / self.batch_size):
            raise IndexError

        batch_start = idx * self.batch_size
        batch_end = min((idx+1) * self.batch_size, self.positions.shape[0])


        # Assume training on GPU

        pos_tensor = torch.from_numpy(
            self.positions[batch_start: batch_end]
        )
        sdf_tensor = torch.from_numpy(
            self.sdf_values[batch_start: batch_end],
        )

        if self.normals is not None:
            normal_tensor = torch.from_numpy(
                self.normals[batch_start: batch_end],
            )
            return pos_tensor, sdf_tensor, normal_tensor
        else:
            return pos_tensor, sdf_tensor, None



class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, input):
        # Reshape into 1D tensor
        x = self.flatten(input)
        return self.model(x)

def train_iteration(
        dataset,
        model,
        loss_fn,
        optimizer,
        loss_dict=None,
        printOut=False
):
    """
        One training iteration
    """
    r_sdf_loss = 0
    r_norm_loss = 0
    r_eik_loss = 0
    r_t_loss = 0

    for batchIdx, (position, sdf_gt, normals_gt) in enumerate(dataset):
        position = position.to('cuda')
        sdf_gt = sdf_gt.to('cuda')

        if normals_gt is not None:
            normals_gt = normals_gt.to('cuda')

        # Prediction

        # Output of field
        field_output = model.get_outputs_raw(position)
        loss, l1, eik, norm, run = loss_fn(field_output, sdf_gt, normals_gt)

        r_sdf_loss += l1
        r_eik_loss += eik
        r_norm_loss += norm
        r_t_loss += run

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if printOut:
        loss_dict['sdf'].append(100.0*r_sdf_loss/len(dataset))
        loss_dict['eik'].append(r_eik_loss/len(dataset))
        loss_dict['norm'].append(r_norm_loss/len(dataset))
        print('L1: {}, Eik: {}, Norm: {}'.format(100.0*r_sdf_loss/len(dataset), r_eik_loss/len(dataset), r_norm_loss/len(dataset)))

if __name__ == '__main__':

    SDF_TRAIN_DATA = '/mnt/hdd/data/sdf_prior/samples_normals.npy'

    # Create dataset
    # Use 2**15 points per batch
    ds = SDFDataset(SDF_TRAIN_DATA, 2 ** 13)

    # Specify a model (on GPU)
    # Later will be SDF field
    # model = NeuralNetwork().to('cuda')

    sdf_field_config = SDFFieldConfig(
        use_grid_feature=True,
        num_layers=2,
        num_layers_color=2,
        hidden_dim=256,
        bias=0.5,
        beta_init=0.3,
        use_appearance_embedding=False,
    )

    # Use same configuration as in model
    aabb = np.array([[-0.3, -0.3, -0.3], [0.3, 0.3, 0.3]])

    model = sdf_field_config.setup(
        aabb=torch.from_numpy(aabb).to('cuda'),
        num_images=16,
        spatial_distortion=SceneContraction()
    ).to('cuda')

    def loss_function(field_output, sdf_gt, normal_gt):

        # Use L1 norm between learned SDF and SDF gt
        sdf_pred = field_output[FieldHeadNames.SDF]
        sdf_loss = F.mse_loss(sdf_pred, sdf_gt)

        # Use eikonal loss
        gradients = field_output[FieldHeadNames.GRADIENT]
        eikonal_loss = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()

        # Use angle as loss for normals
        normal_mask = torch.unsqueeze(torch.sum(torch.abs(normal_gt), dim=-1) >= 5E-6, dim=1)
        normals_pred = field_output[FieldHeadNames.NORMAL]
        normals_pred = F.normalize(normals_pred, p=2, dim=-1)
        normal_gt = F.normalize(normal_gt, p=2, dim=-1)
        # cos = (1.0 - torch.sum(normals_pred * normal_gt * normal_mask, dim=-1)).mean()
        l1_norm = torch.abs((normals_pred - normal_gt)*normal_mask).sum(dim=-1).mean()

        loss = sdf_loss + 0.1 * eikonal_loss + 1.0 * l1_norm

        return loss, sdf_loss.item(), eikonal_loss.item(), l1_norm.item(), loss.item()

    # Train model
    EPOCHS = 50
    loss_dict = dict()
    loss_dict['sdf'] = []
    loss_dict['eik'] = []
    loss_dict['norm'] = []
    loss_dict['t_loss'] = []
    for ep in range(EPOCHS):
        train_iteration(
            ds,
            model,
            loss_fn = loss_function,
            optimizer = torch.optim.Adam(model.parameters(), lr=1E-4, betas=(0.9, 0.999)),
            loss_dict=loss_dict,
            printOut = True)


    import matplotlib.pyplot as plt

    plt.plot(np.linspace(0, 1, EPOCHS), np.array(loss_dict['sdf']), label='L1')
    plt.plot(np.linspace(0, 1, EPOCHS), np.array(loss_dict['eik']), label='Eikonal')
    plt.plot(np.linspace(0, 1, EPOCHS), np.array(loss_dict['norm']), label='Normal')
    plt.legend()
    plt.title('Pretraining loss')
    plt.savefig('out.png')

    MESH_PATH = '/mnt/hdd/extracted/sdf_pretrain/out_pretrain_final.ply'
    MODEL_OUT_PATH = '/mnt/hdd/pretrained/sdf_field.tor'

    mPath = Path(MODEL_OUT_PATH)
    mPath.parent.mkdir(parents=True, exist_ok=True)

    # Save SDF field model
    torch.save(model.state_dict(), str(mPath))

    # Extract mesh of SDF field
    extract_mesh(
        MESH_PATH,
        model,
        256,
        bounding_box_min=(-0.16, -0.313, -0.12),
        bounding_box_max=(0.16, 0.147, 0.12)
    )
