import torch

def get_activations(model, input_tensor, layer_num, verbose=True):
    """
    Get the internal representation of input tensor at after the layer 'layer_num'
    """
    
    layers = list(model.children())
    layer = layers[layer_num]

    if verbose:
        print(f'Checking residues of the input after layer : {layer}')

    # Variable pour stocker les activations
    activations = None

    # Hook pour capturer les activations
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output

    handle = layer.register_forward_hook(hook_fn)
    output = model(input_tensor)
    handle.remove()

    return {'output' : output, 'activations' : activations}

def masked_grads(forget_grads, retain_grads, Masks):
    masks = []
    m_grads = []
    for idx_param, f_grad in enumerate(forget_grads):
        r_grad = retain_grads[idx_param]
        mask = f_grad * r_grad > 0
        if len(Masks) <= idx_param:
            masks.append(mask)
        else:
            masks.append(mask * Masks[idx_param])
        # /!\/!\/!\
        # Formule du gradient pour la descente
        m_grad = masks[idx_param] * f_grad * torch.abs(r_grad)
        # /!\/!\/!\
        m_grads.append(m_grad)
    return m_grads, masks

def update_param(model, forget_grads, retain_grads, Masks = []):
    m_grads, masks = masked_grads(forget_grads, retain_grads, Masks)
    max_perc_param = 0.
    for idx_param, param in enumerate(model.parameters()):
        param.grad = m_grads[idx_param]
        perc_param = 100 * torch.sum(param.grad > 0)/torch.tensor(param.size()).prod()
        if perc_param > max_perc_param:
            max_perc_param = perc_param
    return max_perc_param, masks

def get_grads(model, softmax, forget_input, retain_input, device, verbose=False):
    r_input_tensor, r_label_tensor = retain_input[0].to(device), retain_input[1].to(device)
    # Attention !
    # On a retiré une classe
    r_label_tensor = r_label_tensor - 1 * (r_label_tensor > class_num)
    # ^^^ /!\ ^^^
    f_input_tensor, _ = forget_input[0].to(device), forget_input[1].to(device)

    # get r_grads
    model.zero_grad()
    r_input_tensor.requires_grad = True
    retain_dict = get_activations(model, r_input_tensor, -1, verbose=verbose)

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(retain_dict['output'], r_label_tensor)
    loss.backward()

    r_grads = []
    for param in model.parameters():
        r_grads.append(param.grad)

    # get f_grads
    model.zero_grad()
    f_input_tensor.requires_grad = True
    forget_dict = get_activations(model, f_input_tensor, -1, verbose=verbose)

    r_activations = retain_dict['activations'].detach()
    target = convex_target(softmax, r_activations, r_label_tensor)

    criterion = torch.nn.MSELoss()
    loss = criterion(torch.mean(forget_dict['activations'], dim=0), target)
    loss.backward()

    f_grads = []
    for param in model.parameters():
        f_grads.append(param.grad)
    
    return f_grads, r_grads

def unlearn_step(model, softmax, forget_input, retain_input, optimizer, device, Masks=[]):
    optimizer.zero_grad()
    f_grads, r_grads = get_grads(model, softmax, forget_input, retain_input, verbose=False)
    max_perc_param, masks = update_param(model, f_grads, r_grads, Masks=Masks)
    optimizer.step()
    return max_perc_param, masks

import torch
from torch.func import vmap, grad, functional_call

def get_grad_mean_std(model, criterion, x, y, mini_batch=16):
    training_mode = model.training
    model.eval()

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def compute_loss(params, buffers, x_sample, y_sample):
        output = functional_call(model, (params, buffers), (x_sample.unsqueeze(0),))
        return criterion(output, y_sample.unsqueeze(0))

    compute_grad = grad(compute_loss)
    batch_size = x.shape[0]
    num_splits = batch_size // mini_batch

    mini_batch_grads = {name: [] for name in params}

    for i in range(num_splits):
        x_mini = x[i * mini_batch:(i + 1) * mini_batch]
        y_mini = y[i * mini_batch:(i + 1) * mini_batch]

        batched_grads = vmap(compute_grad, (None, None, 0, 0))(params, buffers, x_mini, y_mini)

        for name, grad_values in batched_grads.items():
            if grad_values.numel() > 0:
                mini_batch_grads[name].append(grad_values.mean(dim=0))
            else:
                print(f"Warning: Gradients vides pour {name} dans le mini-batch {i}")

    gradient_mean = {}
    gradient_std = {}

    for idx, (name, grads) in enumerate(mini_batch_grads.items()):
        if grads:
            stacked_grads = torch.stack(grads, dim=0)
            gradient_mean[idx] = stacked_grads.mean(dim=0)
            gradient_std[idx] = stacked_grads.std(dim=0, unbiased=True)
        else:
            raise ValueError(f"Les gradients pour le paramètre {name} sont vides")

    if training_mode:
        model.train()

    return gradient_mean, gradient_std