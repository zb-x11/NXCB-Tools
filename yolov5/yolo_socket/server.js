const express = require('express');
const cors = require('cors');
const WebSocket = require('ws');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 3000;

// 存储连接的客户端（网页）
const clients = new Set();

// 接收YOLOv5的POST请求
app.post('/detection', (req, res) => {
  const detectionData = req.body;
  console.log('收到检测数据:', detectionData);
  
  // 广播给所有连接的客户端
  broadcastToClients(detectionData);
  
  res.status(200).json({status: 'received'});
});

// 广播数据给所有客户端
function broadcastToClients(data) {
  clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(data));
    }
  });
}

// 创建WebSocket服务器
const wss = new WebSocket.Server({ port: 3001 });

wss.on('connection', (ws) => {
  console.log('新的客户端连接');
  clients.add(ws);
  
  ws.on('close', () => {
    console.log('客户端断开连接');
    clients.delete(ws);
  });
  
  ws.on('error', (error) => {
    console.error('WebSocket错误:', error);
  });
});

// SSE端点（Server-Sent Events）
app.get('/events', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*'
  });
  
  // 发送保持连接的消息
  const keepAlive = setInterval(() => {
    res.write(': keepalive\n\n');
  }, 30000);
  
  req.on('close', () => {
    clearInterval(keepAlive);
  });
});

app.listen(PORT, () => {
  console.log(`服务器运行在 http://localhost:${PORT}`);
  console.log(`WebSocket运行在 ws://localhost:3001`);
});