import React, { useState } from 'react';
import { Trash2, Edit2, Play, Check, X, Square } from 'lucide-react';

const MarkingsList = ({ markings, onRemove, onEdit, onSeek, onPlaySection, playingSection, duration }) => {
  const [editingId, setEditingId] = useState(null);
  const [editValues, setEditValues] = useState({ start: '', end: '' });
  
  const handleEditStart = (marking) => {
    setEditingId(marking.id);
    setEditValues({
      start: marking.start.toFixed(1),
      end: marking.end.toFixed(1)
    });
  };
  
  const handleEditSave = (id) => {
    const start = parseFloat(editValues.start);
    const end = parseFloat(editValues.end);
    
    if (start >= 0 && end <= duration && start < end) {
      onEdit(id, start, end);
      setEditingId(null);
    }
  };
  
  const handleEditCancel = () => {
    setEditingId(null);
    setEditValues({ start: '', end: '' });
  };
  
  const getTotalProblemTime = () => {
    return markings.reduce((total, m) => total + (m.end - m.start), 0);
  };
  
  const getProblemPercentage = () => {
    if (!duration) return 0;
    return ((getTotalProblemTime() / duration) * 100).toFixed(1);
  };
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-4 mt-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-800">
          問題箇所 ({markings.length}箇所)
        </h3>
        <div className="text-sm text-gray-600">
          合計: {formatTime(getTotalProblemTime())} ({getProblemPercentage()}%)
        </div>
      </div>
      
      {markings.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          問題箇所をマークしてください
        </div>
      ) : (
        <div className="space-y-2">
          {markings.map((marking, index) => (
            <div
              key={marking.id}
              className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              {editingId === marking.id ? (
                // 編集モード
                <div className="flex items-center space-x-2 flex-1">
                  <span className="text-sm font-medium">#{index + 1}</span>
                  <input
                    type="number"
                    step="0.1"
                    value={editValues.start}
                    onChange={(e) => setEditValues({...editValues, start: e.target.value})}
                    className="w-20 px-2 py-1 border rounded text-sm"
                    placeholder="開始"
                  />
                  <span className="text-sm">〜</span>
                  <input
                    type="number"
                    step="0.1"
                    value={editValues.end}
                    onChange={(e) => setEditValues({...editValues, end: e.target.value})}
                    className="w-20 px-2 py-1 border rounded text-sm"
                    placeholder="終了"
                  />
                  <button
                    onClick={() => handleEditSave(marking.id)}
                    className="p-1 text-green-600 hover:bg-green-100 rounded"
                  >
                    <Check size={16} />
                  </button>
                  <button
                    onClick={handleEditCancel}
                    className="p-1 text-red-600 hover:bg-red-100 rounded"
                  >
                    <X size={16} />
                  </button>
                </div>
              ) : (
                // 表示モード
                <>
                  <div className="flex items-center space-x-3">
                    <span className="text-sm font-medium text-gray-700">
                      #{index + 1}
                    </span>
                    <span className="text-sm text-gray-600">
                      {formatTime(marking.start)} - {formatTime(marking.end)}
                    </span>
                    <span className="text-xs text-gray-500">
                      ({marking.duration.toFixed(1)}秒)
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-1">
                    <button
                      onClick={() => onPlaySection ? onPlaySection(marking) : onSeek(marking.start)}
                      className={`p-1 rounded transition-colors ${
                        playingSection === marking.id 
                          ? 'text-white bg-blue-600 hover:bg-blue-700' 
                          : 'text-blue-600 hover:bg-blue-100'
                      }`}
                      title="この部分を再生"
                    >
                      {playingSection === marking.id ? <Square size={16} /> : <Play size={16} />}
                    </button>
                    <button
                      onClick={() => handleEditStart(marking)}
                      className="p-1 text-gray-600 hover:bg-gray-200 rounded"
                      title="編集"
                    >
                      <Edit2 size={16} />
                    </button>
                    <button
                      onClick={() => onRemove(marking.id)}
                      className="p-1 text-red-600 hover:bg-red-100 rounded"
                      title="削除"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const formatTime = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

export default MarkingsList;