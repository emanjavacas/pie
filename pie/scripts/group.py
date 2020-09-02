
import click
# Can be run with python -m pie.scripts.group
import pie.utils


@click.group()
def pie_cli():
    """ Group command for Pie """


@pie_cli.command()
@click.option('--model_spec', help="Path to model(s)")
@click.option('--batch_size', type=int, default=50, help="Size of the batch")
@click.option('--device', default='cpu', help="Device to use to run the network")
def webapp(model_spec, batch_size, device):
    """ Run the webapp """
    # Until further version, we should explain what's going on
    print("Not supported anymore, do pip install flask_pie")
    raise Exception("The web version of pie has moved to github.com/"
                    "hipster-philology/flask_pie")


@pie_cli.command()
@click.argument('input_path')
@click.argument('model_spec', type=pie.utils.model_spec)
@click.option('--keep_boundaries', is_flag=True,
              help='Keep boundaries from the original input file')
@click.option('--batch_size', type=int, default=50)
@click.option('--device', default='cpu')
@click.option('--use_beam', is_flag=True, default=False)
@click.option('--beam_width', default=10, type=int)
@click.option('--lower', is_flag=True, help="Treat the input as lower case")
@click.option('--max_sent_len', type=int, default=35,
              help='Split sentences longer than this amount')
@click.option('--vrt', is_flag=True, help='Verticalized input format')
def tag(model_spec, input_path, keep_boundaries, batch_size, device,
        use_beam, beam_width, lower, max_sent_len, vrt):
    """ Tag [INPUT_PATH] with model(s) at [MODEL_SPEC]"""
    import pie.scripts.tag
    pie.scripts.tag.run(
        model_spec, input_path, beam_width, use_beam, keep_boundaries,
        device=device, batch_size=batch_size, lower=lower,
        max_sent_len=max_sent_len, vrt=vrt)


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
@click.option('--train_path', default=None,
              help="File used to compute unknown tokens/targets")
@click.option('--settings', help="Settings file used for training")
@click.option('--batch_size', type=int, default=500)
@click.option('--buffer_size', type=int, default=100000)
@click.option('--device', default='cpu')
@click.option('--model_info', is_flag=True, default=False)
@click.option('--full', is_flag=True, default=False)
@click.option('--confusion', default=False, is_flag=True)
@click.option('--report', default=False, is_flag=True)
@click.option('--markdown', default=False, is_flag=True)
@click.option('--use_beam', is_flag=True, default=False)
@click.option('--beam_width', type=int, default=12)
def evaluate(model_path, test_path, train_path, settings, batch_size,
             buffer_size, device, model_info, full, confusion, report,
             markdown, use_beam, beam_width):
    """ Evaluate [MODEL_PATH] against [TEST_PATH] using [TRAIN_PATH] to compute
    unknown tokens"""
    import pie.scripts.evaluate
    pie.scripts.evaluate.run(
        model_path=model_path, test_path=test_path, train_path=train_path,
        settings=settings, batch_size=batch_size, buffer_size=buffer_size,
        device=device, model_info=model_info, full=full, confusion=confusion,
        report=report, markdown=markdown,
        use_beam=use_beam, beam_width=beam_width)


@pie_cli.command("train")
@click.argument('config_path')
def train(config_path):
    """ Train a model using the file at [CONFIG_PATH]"""
    import pie.scripts.train
    import pie.settings
    pie.scripts.train.run(pie.settings.settings_from_file(config_path))


@pie_cli.command("info")
@click.argument("model_file", type=click.Path(exists=True,
                                              file_okay=True,
                                              dir_okay=False,
                                              readable=True))
def info(model_file):
    from pie.models import BaseModel
    import pprint
    m = BaseModel.load(model_file)
    bar = "=====================\n"
    click.echo(bar+"Settings", color="red")
    pprint.pprint(m._settings)
    click.echo(bar+"Architecture", color="red")
    click.echo(repr(m))


if __name__ == "__main__":
    pie_cli()
