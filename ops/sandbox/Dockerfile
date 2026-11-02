FROM mcr.microsoft.com/playwright/python:v1.58.0-noble
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV HOME=/tmp PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN pip install --no-cache-dir playwright==1.58.0 'pydantic>=2,<3' 'PyYAML>=6,<7'
WORKDIR /tmp
COPY labs/code-sandboxes/malicious_script.py /opt/diagnostic.py
COPY anchor_ai/browser_tools.py anchor_ai/sandbox.py /opt/anchor/anchor_ai/
COPY config/browser_policy.yml config/sandbox.yml /opt/anchor/config/
COPY labs/code-sandboxes/malicious_script.py /opt/anchor/labs/code-sandboxes/malicious_script.py
COPY labs/browser-agents/ /opt/anchor/labs/browser-agents/
USER 65532:65532
ENTRYPOINT ["python", "-I", "/opt/diagnostic.py"]
CMD ["diagnostic"]
