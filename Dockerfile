FROM python:3.8-slim-buster
RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/collab-uniba/stress_awareness.git
WORKDIR /stress_awareness
RUN git checkout interface

RUN pip install -r requirements.txt

EXPOSE 20000

CMD ["python", "visualization_tool.py"]