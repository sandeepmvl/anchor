import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from anchor_ai.customization import load_policy, distillation_decision
from anchor_ai.dataset_quality import audit_distillation
from anchor_ai.training import distillation_manifest


def latency(values):
    return {'mean_ms': float(np.mean(values)), 'p50_ms': float(np.percentile(values, 50)),
            'p95_ms': float(np.percentile(values, 95)), 'samples': len(values)}


def quality(rows, task, predictions, labels):
    truth = [r['labels'][task] for r in rows]
    raw = classification_report(truth, predictions, labels=labels,
                                output_dict=True, zero_division=0)
    return {'ticket_ids': [r['id'] for r in rows], 'predictions': list(predictions),
            'accuracy': float(accuracy_score(truth, predictions)),
            'macro_f1': raw['macro avg']['f1-score'],
            'per_class': {label: raw[label] for label in labels},
            'confusion': {'labels': labels, 'rows': 'human', 'columns': 'predicted',
                          'counts': confusion_matrix(truth, predictions, labels=labels).tolist()}}


class BoundedStudent:
    def __init__(self, models, spec, tenant_id):
        self.models, self.spec, self.tenant_id = models, spec, tenant_id
        self.weak_classes = {task: [] for task in models}

    def classify(self, text, task, tenant_id):
        result = {'task': task, 'prediction': None, 'confidence': None,
                  'fallback': True, 'reason': 'unsupported_task',
                  'refund_authorized': False, 'model_override': None}
        if task not in self.spec['permitted_tasks'] or task not in self.models:
            return result
        if tenant_id != self.tenant_id:
            return dict(result, reason='tenant_boundary')
        model = self.models[task]
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 1800:
            return dict(result, reason='invalid_input')
        if model.named_steps['tfidf'].transform([text]).nnz == 0:
            return dict(result, reason='out_of_scope_vocabulary')
        probabilities = model.predict_proba([text])[0]
        index = int(np.argmax(probabilities))
        label = str(model.classes_[index])
        confidence = float(probabilities[index])
        result.update(prediction=label, confidence=confidence)
        if label not in self.spec['labels'][task]:
            return dict(result, reason='unsupported_label')
        if confidence < self.spec['confidence_threshold']:
            return dict(result, reason='low_confidence')
        if label in self.weak_classes[task]:
            return dict(result, reason='weak_predicted_class')
        return dict(result, fallback=False, reason='accepted_tag_only')


def evaluate(rows, balanced=False, policy=None):
    policy = load_policy() if policy is None else policy
    spec = policy['distillation']
    splits = audit_distillation(rows, spec)
    manifest = distillation_manifest(rows, policy, balanced)
    train, validation, held = (splits[s] for s in ('train', 'validation', 'held_out'))
    models, tasks = {}, {}
    hourly = spec['hourly_usd']
    teacher_cost = float(np.mean([r['teacher_usd'] for r in held]))
    teacher_times = [r['teacher_ms'] for r in held]
    for task, labels in spec['labels'].items():
        model = Pipeline([('tfidf', TfidfVectorizer()),
                          ('classifier', LogisticRegression(
                              class_weight='balanced' if balanced else None,
                              random_state=0, max_iter=200))])
        start = time.perf_counter()
        model.fit([r['input'] for r in train], [r['labels'][task] for r in train])
        fit_seconds = time.perf_counter() - start
        models[task] = model
        # Warm-up excludes loading and cold starts; every sample includes vectorization.
        model.predict([held[0]['input']])
        samples, per_ticket, predictions = [], [], []
        for row in held:
            local = []
            for _ in range(5):
                start = time.perf_counter()
                prediction = str(model.predict([row['input']])[0])
                local.append((time.perf_counter() - start) * 1000)
            samples.extend(local)
            per_ticket.append(float(np.mean(local)))
            predictions.append(prediction)
        teacher_predictions = [r['teacher'][task] for r in held]
        student = quality(held, task, predictions, labels)
        teacher = quality(held, task, teacher_predictions, labels)
        student.update(latency=latency(samples), training={
            'wall_seconds': fit_seconds,
            'estimated_compute_usd': fit_seconds * hourly / 3600},
            estimated_usd_per_ticket=float(np.mean(samples)) * hourly / 3600000,
            train_label_counts={label: sum(r['labels'][task] == label for r in train)
                                for label in labels})
        teacher.update(latency=latency(teacher_times), usd_per_ticket=teacher_cost,
                       evidence_status='constructed_recording_not_measured')
        weak = [label for label in labels if teacher['per_class'][label]['f1-score']
                - student['per_class'][label]['f1-score'] > spec['max_f1_gap']]
        routed = [p in weak for p in predictions]
        combined = [t if use_teacher else s for s, t, use_teacher in
                    zip(predictions, teacher_predictions, routed)]
        route_quality = quality(held, task, combined, labels)
        route_cost = student['estimated_usd_per_ticket'] + sum(
            r['teacher_usd'] for r, use in zip(held, routed) if use) / len(held)
        route_latency = latency([ms + (r['teacher_ms'] if use else 0)
                                 for ms, r, use in zip(per_ticket, held, routed)])
        route_pass = (any(routed) and all(
            teacher['per_class'][label]['f1-score']
            - route_quality['per_class'][label]['f1-score'] <= spec['max_f1_gap']
            for label in labels) and route_cost < teacher_cost
            and route_latency['p95_ms'] < teacher['latency']['p95_ms'])
        route = {'fallback_classes': weak, 'fallback_tickets': sum(routed),
                 'quality': route_quality, 'estimated_usd_per_ticket': route_cost,
                 'estimated_latency': route_latency, 'passes': bool(route_pass)}
        decision = distillation_decision(teacher, student, route,
                                        dict(spec, labels=labels))
        tasks[task] = {'teacher': teacher, 'student': student, 'route': route,
                       'teacher_agreement': float(np.mean([
                           s == t for s, t in zip(predictions, teacher_predictions)])),
                       'validation': quality(validation, task,
                           [str(p) for p in model.predict([r['input'] for r in validation])], labels),
                       'decision': decision}
    student = BoundedStudent(models, spec, train[0]['tenant_id'])
    # Weak-class information is trusted internal evaluator output, not request metadata.
    student.weak_classes = {t: tasks[t]['decision']['weak_classes'] for t in tasks}
    report = {'manifest': manifest, 'tasks': tasks, 'deployment_approved': False,
              'measurement_boundary': 'warm batch-one TF-IDF plus prediction; no HTTP or loading',
              'hourly_usd_assumption': hourly,
              'costs_excluded': ['idle hosting', 'teacher training-label generation',
                                'human review', 'data preparation', 'recurring retraining',
                                'query analysis', 'retrieval', 'policy answering'],
              'teacher_shared_call': {'usd_total': sum(r['teacher_usd'] for r in held),
                                      'note': 'both tags in one call; do not sum task teacher costs'},
              'combined_student_usd_per_ticket': sum(
                  value['student']['estimated_usd_per_ticket'] for value in tasks.values()),
              'runtime_boundary_note': 'confidence and vocabulary checks are guards, not calibrated OOD detection'}
    return report, student
