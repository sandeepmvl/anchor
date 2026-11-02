import json

from pydantic import ConfigDict, ValidationError

try:
    from .contracts import TriageLabels as ExistingTriageLabels
except ImportError:
    from contracts import TriageLabels as ExistingTriageLabels


class TriageLabels(ExistingTriageLabels):
    model_config = ConfigDict(strict=True, extra='forbid')


TRIAGE_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'ticket_triage', 'strict': True,
        'schema': TriageLabels.model_json_schema(),
    },
}


class InvalidModelOutput(Exception):
    def __init__(self, reason, errors=None):
        self.reason = reason
        self.errors = errors
        super().__init__('invalid_model_output')

    def detail(self):
        result = {'code': 'invalid_model_output', 'reason': self.reason}
        if self.errors is not None:
            result['errors'] = self.errors
        return result


def validate_triage(raw):
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        raise InvalidModelOutput('invalid_json') from None
    try:
        return TriageLabels.model_validate(decoded)
    except ValidationError as error:
        safe = [
            {'loc': [e['loc'][0] if e['loc'] and
                     e['loc'][0] in ('category', 'urgency') else '$'],
             'type': e['type']}
            for e in error.errors()
        ]
        raise InvalidModelOutput('schema_violation', safe) from None
