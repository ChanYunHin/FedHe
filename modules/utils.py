



def get_model_mask(args):

    
    model_mask = "Adam"

    if args.homo_flag:
        model_mask = model_mask = "homo_" + model_mask

    
    if args.distorted_data:
        model_mask += "_distorted"

    
    return model_mask
