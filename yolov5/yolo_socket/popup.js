document.addEventListener('DOMContentLoaded', () => {
  const statusDiv = document.getElementById('status');
  const currentInfo = document.getElementById('currentInfo');
  const historyList = document.getElementById('historyList');
  
  // 更新状态显示
  function updateStatus(connected) {
    statusDiv.className = connected ? 'status connected' : 'status disconnected';
    statusDiv.textContent = connected ? '已连接到检测服务器' : '连接断开';
  }
  
  // 更新当前检测信息
  function updateCurrentDetection(data) {
    if (data) {
      currentInfo.innerHTML = `
        人数: <span class="detection-count">${data.count}</span><br>
        时间: ${new Date().toLocaleTimeString()}<br>
        置信度: ${(data.detections[0]?.confidence || 0).toFixed(2)}
      `;
    }
  }
  
  // 加载历史记录
  function loadHistory() {
    chrome.storage.local.get(['detectionHistory'], (result) => {
      const history = result.detectionHistory || [];
      historyList.innerHTML = '';
      
      if (history.length === 0) {
        historyList.innerHTML = '<p>暂无检测记录</p>';
        return;
      }
      
      history.slice(0, 10).forEach(item => {
        const div = document.createElement('div');
        div.className = 'detection-item';
        div.innerHTML = `
          <div class="detection-time">${new Date(item.timestamp).toLocaleString()}</div>
          检测到 <span class="detection-count">${item.count}</span> 人
        `;
        historyList.appendChild(div);
      });
    });
  }
  
  // 测试按钮
  document.getElementById('testBtn').addEventListener('click', () => {
    chrome.runtime.sendMessage({
      type: 'DETECTION_RECEIVED',
      count: 2,
      detections: [{confidence: 0.95}],
      timestamp: new Date().toISOString()
    });
    
    // 显示确认
    alert('测试通知已发送！');
  });
  
  // 清空记录按钮
  document.getElementById('clearBtn').addEventListener('click', () => {
    if (confirm('确定要清空所有检测记录吗？')) {
      chrome.storage.local.set({detectionHistory: []}, () => {
        loadHistory();
      });
    }
  });
  
  // 监听来自background的消息
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'SERVER_STATUS') {
      updateStatus(message.connected);
    } else if (message.type === 'NEW_DETECTION') {
      updateCurrentDetection(message.data);
      loadHistory(); // 刷新历史记录
    }
    sendResponse({received: true});
  });
  
  // 初始加载
  loadHistory();
  
  // 检查连接状态
  chrome.runtime.sendMessage({type: 'CHECK_STATUS'}, (response) => {
    if (response) {
      updateStatus(response.connected);
    }
  });
});