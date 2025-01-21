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

def get_grads(model, softmax, forget_input, retain_input, device=device, verbose=False):
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

def unlearn_step(model, softmax, forget_input, retain_input, optimizer, device=device, Masks=[]):
    optimizer.zero_grad()
    f_grads, r_grads = get_grads(model, softmax, forget_input, retain_input, verbose=False)
    max_perc_param, masks = update_param(model, f_grads, r_grads, Masks=Masks)
    optimizer.step()
    return max_perc_param, masks