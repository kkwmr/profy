// 自動的にダウンロードする従来の関数
export const exportToJSON = (data, filename) => {
  const jsonString = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonString], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  URL.revokeObjectURL(url);
};

// ユーザーディレクトリに自動保存する新しい関数
export const saveToUserDirectory = async (data, filename, username) => {
  try {
    // APIエンドポイントにPOSTリクエストを送信
    const response = await fetch('/api/save-evaluation', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username: username,
        filename: filename,
        data: data
      })
    });

    if (!response.ok) {
      console.error('サーバー保存に失敗しました');
      alert('保存に失敗しました。サーバーが起動していることを確認してください。');
    } else {
      console.log(`評価データを保存しました: ${username}/${filename}`);
    }
  } catch (error) {
    console.error('保存エラー:', error);
    alert('保存に失敗しました。サーバーが起動していることを確認してください。');
  }
};

export const importJSON = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        resolve(data);
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = reject;
    reader.readAsText(file);
  });
};