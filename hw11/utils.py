import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd.functional import jacobian
import mdtraj as md


def diag_indices(n, ndim=2):
    idx = torch.arange(n)
    return (idx,) * ndim


# compute kinetic energy
def e_kin(v, m=1.0):
    ekin = torch.sum(0.5 * m * v * v)
    return ekin


def assign_MBv(n_particles, beta, m: float = 1.0):
    """Assign Maxwell-Boltzmann distributed velocities.

    Parameters
    ----------
    v: velocity array

    beta: 1/ (kB * T)

    m: atomic mass


    Return
    ------
    v:     velocity array

    """

    v = torch.normal(mean=0, std=np.sqrt(1.0 / (beta * m)), size=(n_particles, 3))

    return v


class DataGNN(Dataset):
    """Dataset wrapper for GNNs. Takes positions, energies, forces, and atomic numbers as input."""

    def __init__(
        self,
        positions: torch.tensor,
        energies: torch.tensor,
        forces: torch.tensor,
        atomic_numbers: torch.tensor,
        device,
    ):
        # self.x = torch.from_numpy(x).float().to(device)
        # self.y = torch.from_numpy(y).float().to(device)

        self.positions = positions
        self.energies = energies
        self.forces = forces
        self.atomic_numbers = atomic_numbers
        self.len = self.atomic_numbers.shape[0]

    def __getitem__(self, index: int) -> tuple:
        return (
            self.positions[index],
            self.energies[index],
            self.forces[index],
            self.atomic_numbers[index],
        )

    def __len__(self) -> int:
        return self.len


def split_data_gnn(
    pos_full: torch.tensor,
    energies_full: torch.tensor,
    forces_full: torch.tensor,
    atomic_numbers_full: torch.tensor,
    train_fraction: float,
    device: str,
):
    """Generates three pytorch Datasets (test, train, full)
    given positions, energies and atomic numbers.

    Parameters
    ----------
    pos_full: Atomic positions with shape (n_samples, n_atoms, 3)

    energies_full: Atomic positions with shape (n_samples,1)

    atomic_numbers_full: Atomic numbers with shape (n_samples,1)



    Return
    ------
    Positions will be returned with shape (n_samples * n_atoms, 3)"""

    # define fraction of data used for training
    assert pos_full.shape[0] == energies_full.shape[0]

    n_samples = energies_full.shape[0]

    n_train = int(train_fraction * n_samples)

    n_particles = pos_full.shape[-2]

    # get indices for training and test set
    ids = np.arange(n_samples)
    np.random.shuffle(ids)
    ids_train, ids_test = np.split(ids, [n_train])

    all_data_dist = DataGNN(
        pos_full, energies_full, forces_full, atomic_numbers_full, device
    )
    train_data = DataGNN(
        pos_full[ids_train],
        energies_full[ids_train],
        forces_full[ids_train],
        atomic_numbers_full[ids_train],
        device,
    )
    test_data = DataGNN(
        pos_full[ids_test],
        energies_full[ids_test],
        forces_full[ids_test],
        atomic_numbers_full[ids_test],
        device,
    )

    return all_data_dist, train_data, test_data


def get_batch_dimensions(batch_size, n_particles):
    return torch.repeat_interleave(torch.arange(batch_size), n_particles)


def energies_and_forces_model(model, r, z):
    # print("r: ", r.shape, "z: ", z.shape)

    # print(f"Initial r shape: {r.shape}, z shape: {z.shape}")

    # r has shape [batch_size, n_particles, 3]
    batch_size, n_particles, _ = r.shape

    # Flatten r for the model: [batch_size * n_particles, 3]
    r = r.view(-1, 3).clone().requires_grad_(True)

    # Adjust z shape: [batch_size, n_particles] -> Flatten to [batch_size * n_particles]
    z = z.squeeze(-1).view(-1).unsqueeze(-1)
    # print(f"Adjusted r shape: {r.shape}, z shape: {z.shape}")

    # Create batch dimensions: [batch_size * n_particles]
    batch_dimensions = get_batch_dimensions(batch_size, n_particles)

    # Call the model with the correct shapes
    energies = model.forward(z, r.view(-1, 3), batch_dimensions)  # Ensure r is [num_atoms, 3]
    
    # Create vector vec for gradient computation
    n_samples = energies.size(0)
    vec = torch.zeros((n_samples, 1), device=energies.device)
    vec[:, 0] = 1.0
    
    # print("energies: ", energies)
    # Compute forces by gradient
    forces = -1 * torch.autograd.grad(
        outputs=energies,
        inputs=r,
        grad_outputs=vec,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Reshape forces back to [batch_size, n_particles, 3]
    forces = forces.view(batch_size, n_particles, 3)

    return energies, forces



def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.modules.loss._Loss,
    rho_ene: float,
    rho_force: float,
    device,
    scheduler=None
):
    """
    Basic training loop for a pytorch model.

    Parameters:
    -----------
    model : pytorch model.
    train_loader : pytorch Dataloader containing the training data.
    optimizer: Optimizer for gradient descent.
    criterion: Loss function.

    Example usage:
    -----------

    model = (...) # a pytorch model
    criterion = (...) # a pytorch loss
    optimizer = (...) # a pytorch optimizer
    trainloader = (...) # a pytorch DataLoader containing the training data

    epochs = 10000
    log_interval = 1000
    for epoch in range(1, epochs + 1):
        loss_train = train(model, trainloader,optimizer, criterion)

        if epoch % log_interval == 0:
            print(f'Train Epoch: {epoch} Loss: {loss_train:.6f}')
        losses_train.append(loss_train)


    """

    # Set model to training mode
    model.train()
    epoch_loss = 0

    n_batches = len(train_loader)

    # Loop over each batch from the training set
    for positions, energies, forces, z in train_loader:
        # Copy data to device
        positions = positions.to(device)
        z = z.to(device)
        energies = energies.to(device)
        forces = forces.to(device)

        n_particles = positions.shape[-2]



        # set optimizer to zero grad to remove previous gradients
        optimizer.zero_grad()

        # print("positions shape: ", positions.shape, "z shape: ", z.shape)

        # Pass data through the network
        model_energies, model_forces = energies_and_forces_model(model, positions, z)

        # Calculate loss for energy
        loss_forces = (1 / n_particles) * torch.sum((forces - model_forces) ** 2)

        # Calculate loss for energy
        loss_energies =  torch.sum((model_energies - energies) ** 2 )


        # Total loss
        loss = rho_ene * loss_energies + rho_force * loss_forces

        # get gradients
        loss.backward()

        # gradient descent
        optimizer.step()

        epoch_loss += loss.data.item()

    return epoch_loss / n_batches


def energy_lj(r: torch.tensor, epsilon: float = 1.0, sigma: float = 1.0):
    """Compute the Lennard-Jones energy of a system with positions r

    Parameters
    ----------
    r: atomic configuration

    sigma: LJ potential parameter sigma, default 1.0

    epsilon: LJ potential parameter epsilon, default 1.0

    Return
    ------
    ene: total LJ energy of atomic configuration

    """

    def lj(dist, epsilon: float = 1.0, sigma: float = 1.0):
        return 4 * epsilon * ((sigma / dist) ** 12 - (sigma / dist) ** 6)

    distances = F.pdist(r)

    pair_energies = torch.vmap(lj)(distances)

    return torch.sum(pair_energies)


def get_forces(energy_fn, r: torch.Tensor):
    """Computes the forces acting on a configuration r for a given energy function using backpropagation.

    Parameters
    ----------
    r: atomic configuration

    energy_fn: energy function


    Return
    ------
    f: atomic forces
    """

    # Create a new tensor that requires grad for force computation.
    # This avoids that the computation graph for all potential energies is stored.
    r_with_grad = torch.clone(r)
    r_with_grad.requires_grad = True

    # evaluate energy
    energy = energy_fn(r_with_grad)

    # compute energy gradients w.r.t. r (i.e. de / dr)
    de_dr = torch.autograd.grad(
        energy, r_with_grad, grad_outputs=None, create_graph=False
    )

    # forces are given by negative gradient of energy w.r.t. atomic positions
    forces = -de_dr[0]

    return forces


def radial_distribution_function(pos, n_bins=200, n_dims=3, r_range=None, **kwargs):
    """Compute the RDF of the data, using mdtraj"""
    n_samples = pos.shape[0]
    num_particles = pos.shape[1]
    top = md.Topology()
    res = top.add_residue("LJ", top.add_chain())
    for i in range(num_particles):  # element type does not matter
        top.add_atom("Ar", md.element.Element.getBySymbol("Ar"), res)

    box = 500 * np.ones((n_samples, 3))
    # assert jnp.abs(data.max()) <= 1 and data.min() >= 0, "data should be rescaled"
    assert pos.shape[-1] == 3 and pos.shape[-2] == num_particles

    if r_range is None:
        # r_range = (0, np.sqrt(np.sum(np.power(box, 2))) / 2)
        r_range = (0, 5)

    # box is needed for PBC. assumed to be cubic/squared
    unitcell = {
        "unitcell_lengths": box,
        "unitcell_angles": np.full((pos.shape[0], n_dims), 90),
    }
    # create mdtraj traj object
    traj = md.Trajectory(pos, top, **unitcell)
    ij = np.array(np.triu_indices(num_particles, k=1)).T
    rdf = md.compute_rdf(traj, ij, r_range=r_range, n_bins=n_bins)

    return rdf


def visualize_graph(positions, edge_ids, dim_1, dim_2):
    plt.figure()
    for e_id in edge_ids.t():
        plt.plot(
            [positions[e_id[0], dim_1], positions[e_id[1], dim_1]],
            [positions[e_id[0], dim_2], positions[e_id[1], dim_2]],
            "r-",
            zorder=0,
        )
    plt.scatter(positions[:, 0], positions[:, 1], s=50)


def langevin_step(
    r: torch.tensor,
    v: torch.tensor,
    f: torch.tensor,
    energy_fn: callable,
    dt: float,
    lc: float,
):
    """Langevin step: MD in NVT ensemble.

    Parameters
    ----------
    r: positions

    v: velocities

    f: forces

    dt: time step

    lc: Parameters Langevin dynamics

    energy_fn: energy function

    Return
    ------
    r: updated position

    v: updated velocities

    f: updated forces


    """
    com = torch.mean(v, axis=0)
    v -= com

    # langevin thermo, 1st half step
    v = lc[0] * v + lc[1] * torch.normal(mean=0.0, std=1.0, size=(v.shape))

    # Verlet part of MD step
    v = v + 0.5 * f * dt

    # update positions
    r = r + v * dt

    f = get_forces(energy_fn, r)

    v = v + 0.5 * f * dt

    # langevin thermo, 2nd half step
    v = lc[0] * v + lc[1] * torch.normal(mean=0.0, std=1.0, size=(v.shape))

    return r, v, f




class MLP(nn.Module):
    def __init__(self, n_units: list, activation=nn.ReLU()):
        """
        Simple multi-layer perceptron (MLP).

        Parameters:

        -----------
        n_units : List of integers specifying the dimensions of input and output and the hidden layers.
        activation: Activation function used for non-linearity.

        Example:
        -----------

        dim_hidden = 100
        dim_in = 2
        dim_out = 5

        # MLP with input dimension 2, output dimension 5, and 4 hidden layers of dimension 100
        model = MLP([dim_in,
                    dim_hidden,
                    dim_hidden,
                    dim_hidden,
                    dim_hidden,
                    dim_out],activation=nn.ReLU()).to(DEVICE)

        """
        super().__init__()

        # Get input and output dimensions
        dims_in = n_units[:-1]
        dims_out = n_units[1:]

        layers = []

        # Add linear layers (and activation function after all layers except the final one)
        for i, (dim_in, dim_out) in enumerate(zip(dims_in, dims_out)):
            layers.append(torch.nn.Linear(dim_in, dim_out))

            if i < len(n_units) - 2:
                layers.append(activation)

        self._layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)

    def count_parameters(self):
        """
        Counts the number of trainable parameters.

        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
