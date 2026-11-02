FROM python:3.11-slim
WORKDIR /tmp
COPY labs/code-sandboxes/malicious_script.py /opt/diagnostic.py
USER 65532:65532
ENTRYPOINT ["python", "-I", "/opt/diagnostic.py"]
CMD ["diagnostic"]
