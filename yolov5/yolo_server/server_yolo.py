# yolo_server/server.py
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from datetime import datetime
import threading
import time
import json
import base64
import os

app = Flask(__name__)
CORS(app)  # 允许跨域

# 存储检测数据
detection_history = []
connected_clients = []

# HTML监控页面
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>YOLO检测监控</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            padding: 30px;
            border-radius: 15px 15px 0 0;
            text-align: center;
            margin-bottom: 20px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
            margin: 10px 0;
        }
        .detection-log {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-top: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .log-entry {
            border-left: 5px solid #4CAF50;
            padding: 15px;
            margin: 15px 0;
            background: #f9f9f9;
            border-radius: 0 8px 8px 0;
        }
        .log-time {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        .log-count {
            color: #2196F3;
            font-weight: bold;
            font-size: 1.2em;
        }
        .person-info {
            background: #e8f5e9;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            font-size: 0.9em;
        }
        .status-connected {
            color: #4CAF50;
            font-weight: bold;
        }
        .status-disconnected {
            color: #f44336;
            font-weight: bold;
        }
        .realtime-notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 1000;
            animation: slideIn 0.5s ease-out;
            display: none;
        }
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 YOLOv5实时检测监控系统</h1>
        <p>接收并显示YOLOv5的检测数据</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div>总检测次数</div>
            <div class="stat-value">{{ total_detections }}</div>
        </div>
        <div class="stat-card">
            <div>在线客户端</div>
            <div class="stat-value">{{ client_count }}</div>
        </div>
        <div class="stat-card">
            <div>服务器端口</div>
            <div class="stat-value">3000</div>
        </div>
        <div class="stat-card">
            <div>服务器状态</div>
            <div class="stat-value status-connected">运行中</div>
        </div>
    </div>

    <div class="detection-log">
        <h2>📊 最近检测记录</h2>
        {% if detection_history %}
            {% for log in detection_history[:10] %}
            <div class="log-entry">
                <div class="log-time">{{ log.timestamp }}</div>
                <div>检测到 <span class="log-count">{{ log.count }}</span> 人</div>
                {% if log.detections %}
                    {% for det in log.detections %}
                    <div class="person-info">
                        👤 {{ det.class_name or 'person' }} - 置信度: {{ "%.2f"|format(det.confidence) }}
                        {% if det.bbox %}
                        - 位置: ({{ "%.0f"|format(det.bbox.x1) }}, {{ "%.0f"|format(det.bbox.y1) }})
                        {% endif %}
                    </div>
                    {% endfor %}
                {% endif %}
                {% if log.source %}
                    <div style="font-size: 0.8em; color: #888; margin-top: 5px;">
                        来源: {{ log.source.path or '未知' }}
                    </div>
                {% endif %}
            </div>
            {% endfor %}
        {% else %}
            <div style="text-align: center; padding: 40px; color: #999;">
                <h3>⏳ 等待检测数据...</h3>
                <p>启动YOLOv5后，检测数据将显示在这里</p>
            </div>
        {% endif %}
    </div>

    <div style="background: white; border-radius: 10px; padding: 20px; margin-top: 30px;">
        <h3>📝 使用说明</h3>
        <ol style="margin-top: 15px; padding-left: 20px;">
            <li><strong>启动YOLOv5检测：</strong><br>
                <code>python detect.py --weights yolov5s.pt --source 0 --webhook-enabled --view-img</code>
            </li>
            <li><strong>测试服务器连接：</strong><br>
                访问 <a href="/test" target="_blank">http://localhost:3000/test</a>
            </li>
            <li><strong>实时通知：</strong><br>
                检测到人时，Chrome插件会收到通知
            </li>
        </ol>
    </div>

    <div id="notification" class="realtime-notification">
        🎯 检测到人员！
    </div>

    <script>
        // 自动刷新页面
        setTimeout(() => {
            location.reload();
        }, 10000); // 每10秒刷新

        // 测试服务器连接
        fetch('/test')
            .then(response => response.json())
            .then(data => {
                console.log('服务器连接正常:', data);
            })
            .catch(error => {
                console.error('服务器连接失败:', error);
            });

        // 显示实时通知
        function showNotification(count) {
            const notification = document.getElementById('notification');
            notification.innerHTML = `🎯 检测到 ${count} 人！`;
            notification.style.display = 'block';

            setTimeout(() => {
                notification.style.display = 'none';
            }, 5000);
        }

        // 使用EventSource接收服务器推送
        const eventSource = new EventSource('/events');

        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'DETECTION') {
                showNotification(data.count);
                // 刷新页面显示最新数据
                setTimeout(() => {
                    location.reload();
                }, 1000);
            }
        };

        eventSource.onerror = function(error) {
            console.error('EventSource错误:', error);
        };
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """监控页面"""
    return render_template_string(HTML_TEMPLATE,
                                  total_detections=len(detection_history),
                                  client_count=len(connected_clients),
                                  detection_history=detection_history[::-1])  # 最新的在前面


@app.route('/detection', methods=['POST', 'OPTIONS'])
def handle_detection():
    """接收YOLOv5的检测数据"""
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.json
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print('\n' + '=' * 60)
        print(f'✅ [{timestamp}] 收到YOLOv5检测数据')
        print(f'   检测人数: {data.get("count", 0)}')
        print(f'   时间: {data.get("timestamp", "未知")}')

        if data.get('detections'):
            for i, det in enumerate(data['detections']):
                print(f'   检测{i + 1}: {det.get("class_name", "person")} - 置信度: {det.get("confidence", 0):.2f}')

        print('=' * 60)

        # 保存到历史记录
        history_entry = {
            'timestamp': timestamp,
            'count': data.get('count', 0),
            'detections': data.get('detections', []),
            'source': data.get('source', {})
        }

        detection_history.append(history_entry)

        # 限制历史记录大小
        if len(detection_history) > 100:
            detection_history.pop(0)

        # 通知所有连接的客户端
        notify_clients({
            'type': 'DETECTION',
            'timestamp': timestamp,
            'count': data.get('count', 0),
            'detections': data.get('detections', [])
        })

        return jsonify({
            'status': 'success',
            'message': f'收到{data.get("count", 0)}人检测数据',
            'timestamp': timestamp
        })

    except Exception as e:
        print(f'❌ 处理请求时出错: {e}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/test')
def test():
    """测试接口"""
    return jsonify({
        'status': 'success',
        'message': '服务器运行正常',
        'timestamp': datetime.now().isoformat(),
        'detection_count': len(detection_history),
        'client_count': len(connected_clients),
        'endpoints': {
            'POST /detection': '接收YOLOv5检测数据',
            'GET /history': '获取历史数据',
            'GET /events': '服务器推送事件',
            'GET /': '监控页面'
        }
    })


@app.route('/history')
def get_history():
    """获取历史数据"""
    return jsonify({
        'status': 'success',
        'count': len(detection_history),
        'history': detection_history[-20:]  # 返回最近20条
    })


@app.route('/events')
def events():
    """服务器推送事件（SSE）"""

    def generate():
        # 发送初始连接消息
        yield f"data: {json.dumps({'type': 'CONNECTED', 'message': '连接成功'})}\n\n"

        # 保持连接
        while True:
            time.sleep(30)
            yield f"data: {json.dumps({'type': 'HEARTBEAT', 'time': datetime.now().isoformat()})}\n\n"

    return app.response_class(generate(), mimetype='text/event-stream')


def notify_clients(data):
    """通知所有客户端"""
    # 这里可以扩展WebSocket功能
    pass


def start_server():
    """启动服务器"""
    print('\n' + '=' * 60)
    print('🚀 启动YOLOv5检测服务器 (Python Flask)')
    print('=' * 60)
    print('📡 服务器地址: http://localhost:3000')
    print('📊 监控页面: http://localhost:3000')
    print('⏰ 启动时间:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('-' * 60)
    print('📝 可用接口:')
    print('  GET  /        - 监控页面')
    print('  POST /detection - 接收YOLOv5数据')
    print('  GET  /history   - 获取历史数据')
    print('  GET  /test      - 测试连接')
    print('  GET  /events    - 服务器推送')
    print('-' * 60)
    print('🔄 等待YOLOv5检测数据...')
    print('🛑 按 Ctrl+C 停止服务器')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    start_server()
    app.run(host='0.0.0.0', port=3000, debug=False)