FROM python:3.8
# RUN apt-get update && apt-get install -y git
# RUN git clone https://github.com/collab-uniba/stress_awareness.git
# RUN git checkout interface

RUN pip install -r requirements.txt

COPY . /stress_awareness
WORKDIR /stress_awareness


EXPOSE 20000

ENV DISPLAY=:0
ENTRYPOINT ["python", "visualization_tool.py"]