import React, { useRef } from 'react';
import { Upload, Music } from 'lucide-react';

const FileManager = ({ onFileLoad }) => {
  const fileInputRef = useRef(null);
  
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = Array.from(e.dataTransfer.files).filter(
      file => file.type.startsWith('audio/')
    );
    
    if (files.length > 0) {
      onFileLoad(files);
    }
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  
  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      onFileLoad(files);
    }
  };
  
  return (
    <div className="max-w-2xl mx-auto mt-12">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-blue-500 transition-colors cursor-pointer"
        onClick={() => fileInputRef.current?.click()}
      >
        <Music size={48} className="mx-auto text-gray-400 mb-4" />
        <h2 className="text-xl font-semibold text-gray-700 mb-2">
          音声ファイルを選択
        </h2>
        <p className="text-gray-500 mb-4">
          クリックして選択、またはドラッグ＆ドロップ
        </p>
        <p className="text-sm text-gray-400">
          対応形式: MP3, WAV, M4A, OGG
        </p>
        
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          multiple
          onChange={handleFileSelect}
          className="hidden"
        />
        
        <button className="mt-6 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors inline-flex items-center space-x-2">
          <Upload size={20} />
          <span>ファイルを選択</span>
        </button>
      </div>
      
      <div className="mt-8 p-4 bg-gray-50 rounded-lg">
        <h3 className="font-semibold text-gray-700 mb-2">使い方</h3>
        <ol className="text-sm text-gray-600 space-y-1">
          <li>1. 音声ファイルを選択（複数可）</li>
          <li>2. 波形上でドラッグして問題箇所をマーク</li>
          <li>3. スライダーで点数を設定</li>
          <li>4. 「完了して次へ」で保存＆次の音声へ</li>
        </ol>
      </div>
    </div>
  );
};

export default FileManager;