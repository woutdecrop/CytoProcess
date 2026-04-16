from cytoprocess.commands import overwrite_ecotaxa, prepare, upload


def run(ctx, project, username: str | None = None, password: str | None = None):
    prepare.run(ctx, project, force=True, include_predictions=False)
    upload.run(ctx, project, username=username, password=password, update=False)
    overwrite_ecotaxa.run(ctx, project, username=username, password=password)
