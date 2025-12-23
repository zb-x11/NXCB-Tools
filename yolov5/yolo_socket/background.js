// 监听来自网页的消息
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'DETECTION_RECEIVED') {
    // 显示浏览器通知
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icon.png',
      title: `检测到 ${request.count} 人`,
      message: `置信度: ${request.detections[0]?.confidence?.toFixed(2) || 'N/A'}`,
      priority: 2
    });
    
    // 保存到存储
    chrome.storage.local.get(['detectionHistory'], (result) => {
      const history = result.detectionHistory || [];
      history.unshift({
        timestamp: new Date().toISOString(),
        count: request.count,
        detections: request.detections
      });
      
      // 只保留最近100条记录
      chrome.storage.local.set({
        detectionHistory: history.slice(0, 100)
      });
    });
  }
  
  sendResponse({status: 'OK'});
});

// 创建本地服务器监听
const serverPort = 3000;

// 与内容脚本通信
function sendToContentScript(message) {
  chrome.tabs.query({}, (tabs) => {
    tabs.forEach(tab => {
      chrome.tabs.sendMessage(tab.id, message);
    });
  });
}

// 模拟接收webhook（在实际使用中，你需要一个真正的服务器）
chrome.runtime.onInstalled.addListener(() => {
  console.log('YOLO Detection Monitor installed');
});