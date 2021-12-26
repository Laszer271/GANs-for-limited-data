def cast_list(el):
    return el if isinstance(el, list) else [el]


def train(num_image_tiles=None, multi_gpus=False, data=None,
          load_from=None, new=True, num_train_steps=None, 
          name=None, seed=42, **kwargs):
    if num_image_tiles is None:  # default
        if kwargs['image_size'] <= 512:
            num_images_tiles = 8
        else:
            num_image_tiles = 4
    kwargs['num_image_tiles'] = num_images_tiles
    
    model_args = {key: kwargs[key] for key in kwargs.keys() if key in MODEL_ARGS}
    model_args['name'] = name
    
    unused_args = {key: kwargs[key] for key in kwargs.keys() if key not in MODEL_ARGS}
    not_found_args = [arg for arg in MODEL_ARGS if arg not in model_args.keys()]
    print('Unused:', unused_args)
    print('Not found:', not_found_args)
    
    #cli.run_training(0, 1, model_args, data, load_from, new, num_train_steps, name, seed)

    model = Trainer(**model_args)

    if not new:
        model.load(load_from)
    else:
        model.clear()

    model.set_data_src(data)

    progress_bar = tqdm(initial = model.steps, total = num_train_steps, mininterval=10., desc=f'{name}<{data}>')
    while model.steps < num_train_steps:
        retry_call(model.train, tries=3, exceptions=NanException)
        progress_bar.n = model.steps
        progress_bar.refresh()
        if model.steps % 50 == 0:
            model.print_log()

    model.save(model.checkpoint_num)

train(**cli_d)
#cli.train_from_folder(**cli_d)