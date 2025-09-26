import React, { useEffect, useState } from 'react';

const EvaluationHistory = ({ username, onLoad }) => {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!username) return;
    setLoading(true);
    fetch(`/api/evaluations/${username}`)
      .then(r => r.json())
      .then(json => {
        setItems(json.evaluations || []);
      })
      .catch(e => {
        console.error(e);
        setError('評価データの取得に失敗しました');
      })
      .finally(() => setLoading(false));
  }, [username]);

  const handleLoad = async (filename) => {
    try {
      const r = await fetch(`/api/evaluation/${username}/${encodeURIComponent(filename)}`);
      const json = await r.json();
      if (!json.ok) throw new Error('not ok');
      onLoad(json.data, filename);
    } catch (e) {
      console.error(e);
      alert('評価データの読み込みに失敗しました');
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700">評価履歴（{username}）</h3>
        {loading && <span className="text-xs text-gray-500">読み込み中...</span>}
      </div>
      {error && <div className="text-sm text-red-600 mb-2">{error}</div>}
      {items.length === 0 ? (
        <div className="text-sm text-gray-500">保存済みの評価はありません</div>
      ) : (
        <div className="max-h-48 overflow-y-auto divide-y">
          {items.map((it) => (
            <div key={it.filename} className="py-2 flex items-center justify-between">
              <div className="min-w-0 mr-2">
                <div className="text-sm text-gray-800 truncate">{it.audioFile || '(不明な音声)'} — {it.filename}</div>
                <div className="text-xs text-gray-500">
                  {new Date(it.timestamp).toLocaleString()} / Score: {typeof it.score === 'number' ? it.score : '-'}
                </div>
              </div>
              <button
                className="text-blue-600 hover:text-blue-800 text-sm whitespace-nowrap"
                onClick={() => handleLoad(it.filename)}
              >読み込み</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default EvaluationHistory;

