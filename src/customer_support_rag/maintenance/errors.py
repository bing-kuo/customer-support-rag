"""Safe failures for the knowledge maintenance lifecycle."""


class MaintenanceError(Exception):
    """A maintenance step cannot continue.

    Messages name a category. They do not include document text, approval
    evidence, database paths, or hashes of secrets.
    """
