#!/bin/bash



# 启动 FastAPI 服务
uvicorn mytools.geo_locate_servicev2:app --host 0.0.0.0 --port 8000 --reload

