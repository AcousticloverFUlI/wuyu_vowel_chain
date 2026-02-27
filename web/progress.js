// 伪代码结构参考
import { LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';

const ThesisDashboard = () => {
  return (
    <div className="bg-gray-50 min-h-screen p-8 font-sans">
      {/* 顶部进度条 */}
      <header className="mb-10">
        <h1 className="text-2xl font-serif text-slate-800">吴语音韵演变研究 - 进度看板</h1>
        <div className="w-full bg-gray-200 h-2 mt-4 rounded-full">
          <div className="bg-blue-600 h-2 rounded-full" style={{ width: '65%' }}></div>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {/* 第一栏：数据 */}
        <div className="bg-white p-6 rounded-xl shadow-sm">
          <h2 className="text-lg font-bold mb-4">数据板块</h2>
          <LineChart width={300} height={200} data={forestData}>
            <XAxis dataKey="date" />
            <YAxis />
            <Line type="monotone" dataKey="hours" stroke="#2563eb" />
          </LineChart>
          {/* 方言点状态列表 */}
          <ul className="mt-4 space-y-2">
            {dialectPoints.map(p => (
              <li key={p.id} className="flex justify-between text-sm">
                <span>{p.name}</span>
                <span className="text-blue-500 font-medium">{p.status}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* 第二栏：算法笔记 */}
        <div className="bg-white p-6 rounded-xl shadow-sm">
          <h2 className="text-lg font-bold mb-4">算法模型思考</h2>
          <div className="prose prose-sm">
             {/* 这里渲染 Markdown */}
             <p>关于元音演变的链移模型分析...</p>
          </div>
        </div>

        {/* 第三栏：会议预定 */}
        <div className="bg-white p-6 rounded-xl shadow-sm">
          <h2 className="text-lg font-bold mb-4">会议日程</h2>
          <div className="space-y-4">
             {/* 简单的日程卡片 */}
             <div className="border-l-4 border-blue-500 pl-3">
               <p className="text-xs text-gray-400">2026-03-01</p>
               <p className="text-sm font-medium">双周汇报：元音格局图初步分析</p>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};