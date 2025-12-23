// 监听来自background脚本的消息
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'DETECTION_ALERT') {
    displayDetectionAlert(message.data);
  }
  sendResponse({received: true});
});

// 在网页上显示检测警报
function displayDetectionAlert(data) {
  // 创建或更新通知元素
  let alertDiv = document.getElementById('yolo-detection-alert');
  if (!alertDiv) {
    alertDiv = document.createElement('div');
    alertDiv.id = 'yolo-detection-alert';
    alertDiv.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: rgba(255, 59, 48, 0.9);
      color: white;
      padding: 15px;
      border-radius: 8px;
      z-index: 10000;
      font-family: Arial, sans-serif;
      max-width: 300px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      animation: slideIn 0.3s ease-out;
    `;
    
    const style = document.createElement('style');
    style.textContent = `
      @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }
    `;
    document.head.appendChild(style);
    document.body.appendChild(alertDiv);
  }
  
  alertDiv.innerHTML = `
    <strong>🚨 检测到人员！</strong>
    <p>人数: ${data.count}</p>
    <p>时间: ${new Date().toLocaleTimeString()}</p>
    <button id="close-alert" style="
      background: white;
      color: #ff3b30;
      border: none;
      padding: 5px 10px;
      border-radius: 4px;
      margin-top: 8px;
      cursor: pointer;
    ">关闭</button>
  `;
  
  document.getElementById('close-alert').onclick = () => {
    alertDiv.style.display = 'none';
  };
  
  // 5秒后自动隐藏
  setTimeout(() => {
    if (alertDiv.style.display !== 'none') {
      alertDiv.style.animation = 'slideOut 0.3s ease-out';
      setTimeout(() => {
        alertDiv.style.display = 'none';
      }, 300);
    }
  }, 5000);
}

// 监听服务器事件（使用EventSource或WebSocket）
function connectToDetectionServer() {
  const eventSource = new EventSource('http://localhost:3000/events');
  
  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    chrome.runtime.sendMessage({
      type: 'DETECTION_RECEIVED',
      ...data
    });
    
    // 在网页上显示
    displayDetectionAlert(data);
  };
  
  eventSource.onerror = (error) => {
    console.error('EventSource failed:', error);
    // 重连机制
    setTimeout(connectToDetectionServer, 5000);
  };
}

// 启动连接
connectToDetectionServer();