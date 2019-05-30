
# Can be run with python -m pie.scripts.group
import pie.utils
import pie.settings

import click


@click.group()
def pie_cli():
    """ Group command for Pie """


@pie_cli.command()
@click.option('--model_spec', help="Path to model(s)")
@click.option('--batch_size', type=int, default=50, help="Size of the batch")
@click.option('--device', default='cpu', help="Device to use to run the network")
def webapp(model_spec, batch_size, device):
    """ Run the webapp """
    import pie.scripts.app
    app = pie.scripts.app.run(device=device, batch_size=batch_size, model_file=model_spec)
    print(
        """
This application should not be used in production.
The webserver given here is only for development purposes
""")
    app.run()


@pie_cli.command()
@click.argument('model_spec', type=pie.utils.model_spec)
@click.argument('input_path')
@click.option('--batch_size', type=int, default=50)
@click.option('--device', default='cpu')
@click.option('--use_beam', is_flag=True, default=False)
@click.option('--beam_width', default=10, type=int)
@click.option('--lower', is_flag=True, help="Treat the input as lower case")
def tag(model_spec, input_path, device, batch_size, lower, beam_width, use_beam):
    """ Tag [INPUT_PATH] with model(s) at [MODEL_SPEC]"""
    import pie.scripts.tag
    pie.scripts.tag.run(
        model_spec, input_path, device, batch_size, lower, beam_width, use_beam)


@pie_cli.command("tag-pipe")
@click.argument('model_spec', type=pie.utils.model_spec)
@click.option('--batch_size', type=int, default=50)
@click.option('--device', default='cpu')
@click.option('--use_beam', is_flag=True, default=False)
@click.option('--beam_width', default=10, type=int)
@click.option('--lower', is_flag=True, help="Lowercase input to tagger", default=False)
@click.option('--tokenize', is_flag=True, help="Tokenize the input", default=False)
def tag_pipe(model_spec, device, batch_size, lower, beam_width, use_beam, tokenize):
    """ Tag the terminal input with [MODEL_SPEC]"""
    import pie.scripts.tag_pipe
    pie.scripts.tag_pipe.run(
        model_spec=model_spec, device=device, batch_size=batch_size,
        lower=lower, beam_width=beam_width, use_beam=use_beam, tokenize=tokenize)


@pie_cli.command("eval")
@click.argument('model_path')
@click.argument('test_path', nargs=-1)
@click.argument('train_path')
@click.option('--settings', help="Settings file used for training")
@click.option('--batch_size', type=int, default=500)
@click.option('--buffer_size', type=int, default=100000)
@click.option('--device', default='cpu')
@click.option('--model_info', is_flag=True, default=False)
@click.option('--full', is_flag=True, default=False)
@click.option('--confusion', default=False, is_flag=True)
def evaluate(model_path, test_path, train_path, settings, batch_size,
             buffer_size, device, model_info, full, confusion):
    """ Evaluate [MODEL_PATH] against [TEST_PATH] using [TRAIN_PATH] to compute
    unknown tokens"""
    import pie.scripts.evaluate
    pie.scripts.evaluate.run(
        model_path=model_path, test_path=test_path, train_path=train_path,
        settings=settings, batch_size=batch_size, buffer_size=buffer_size,
        device=device, model_info=model_info, full=full, confusion=confusion)


@pie_cli.command("train")
@click.argument('config_path')
def train(config_path):
    """ Train a model using the file at [CONFIG_PATH]"""
    import pie.scripts.train
    pie.scripts.train.run(pie.settings.settings_from_file(config_path))


@pie_cli.command("optimize")
@click.argument('config_path')
@click.argument('opt_path')
@click.argument('n_iter', default=20, type=int)
def train(config_path, opt_path, n_iter):
    """ Train a model using the file at [CONFIG_PATH]"""
    import pie.scripts.opt
    pie.scripts.opt.run(
        pie.settings.settings_from_file(config_path),
        pie.scripts.opt.read_opt(opt_path),
        n_iter)


if __name__ == "__main__":
    pie_cli()
