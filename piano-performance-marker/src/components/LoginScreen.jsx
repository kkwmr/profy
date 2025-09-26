import React, { useState } from 'react';
import { User } from 'lucide-react';

const LoginScreen = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [error, setError] = useState('');
  const [isChecking, setIsChecking] = useState(false);
  const [existingUser, setExistingUser] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (username.trim()) {
      // 英数字とアンダースコアのみ許可
      if (!/^[a-zA-Z0-9_]+$/.test(username)) {
        setError('ユーザー名は英数字とアンダースコアのみ使用できます');
        return;
      }
      
      setIsChecking(true);
      try {
        // 既存ユーザーかチェック
        const response = await fetch(`/api/last-evaluation/${username.trim()}`);
        const data = await response.json();
        
        if (data.exists) {
          setExistingUser(data);
          // 確認メッセージを表示（再開 or 別名で新規）
          const message = `ユーザー名「${username}」は既に使用されています。` +
            `\n保存済みデータ: ${data.totalEvaluations} 件` +
            (data.lastAudioIndex ? `\n最後に評価した音声番号: ${data.lastAudioIndex}` : '') +
            `\n\nOK: 同じユーザーで続きから再開` +
            `\nキャンセル: 新規登録するので別のユーザー名を入力`;

          if (window.confirm(message)) {
            onLogin(username.trim(), data.lastAudioIndex);
          } else {
            setExistingUser(null);
          }
        } else {
          // 新規ユーザー（ローカルの進捗キーがあればクリア）
          try { localStorage.removeItem(`${username.trim()}_lastAudioIndex`); } catch {}
          onLogin(username.trim(), null);
        }
      } catch (error) {
        console.error('ユーザーチェックエラー:', error);
        // エラーが発生してもログインは許可
        onLogin(username.trim(), null);
      } finally {
        setIsChecking(false);
      }
    } else {
      setError('ユーザー名を入力してください');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit(e);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center px-4">
      <div className="max-w-md w-full">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
              <User size={32} className="text-blue-600" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900">Piano Performance Marker</h1>
            <p className="text-gray-600 mt-2">評価を開始するにはログインしてください</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-2">
                ユーザー名
              </label>
              <input
                type="text"
                id="username"
                value={username}
                onChange={(e) => {
                  setUsername(e.target.value);
                  setError('');
                }}
                onKeyPress={handleKeyPress}
                className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors ${
                  error ? 'border-red-500' : 'border-gray-300'
                }`}
                placeholder="例: evaluator_1"
                autoFocus
              />
              {error && (
                <p className="mt-2 text-sm text-red-600">{error}</p>
              )}
            </div>

            <button
              type="submit"
              disabled={isChecking}
              className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isChecking ? 'チェック中...' : 'ログイン'}
            </button>
          </form>

          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600">
              <strong>注意事項:</strong>
            </p>
            <ul className="mt-2 text-sm text-gray-600 space-y-1">
              <li>• ユーザー名は英数字とアンダースコアのみ</li>
              <li>• 評価データは自動的に保存されます</li>
              <li>• ユーザーごとに専用フォルダが作成されます</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginScreen;
