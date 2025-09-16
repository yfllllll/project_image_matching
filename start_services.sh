#!/bin/bash  
# 启动地理定位服务  
  
echo "启动地理定位服务..."  
cd /path/to/your/project  
  
# 启动后端服务  
python test.py &  
BACKEND_PID=$!  
  
# 等待后端启动  
sleep 5  
  
# 检查后端健康状态  
curl -f http://localhost:7860/health || {  
    echo "后端服务启动失败"  
    kill $BACKEND_PID  
    exit 1  
}  
  
echo "后端服务启动成功 (PID: $BACKEND_PID)"  
  
# 启动前端服务  
python web_of_test.py &  
FRONTEND_PID=$!  
  
echo "前端服务启动成功 (PID: $FRONTEND_PID)"  
echo "服务已启动完成"  
echo "后端: http://localhost:7860"  
echo "前端: http://localhost:7861"  
  
# 保存PID以便后续停止  
echo $BACKEND_PID > backend.pid  
echo $FRONTEND_PID > frontend.pid