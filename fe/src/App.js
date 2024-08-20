// fe/src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom';
import HomePage from './pages/HomePage';
import MyPage from './pages/MyPage';
import SearchBar from './components/SearchBar';
import './App.css';

// AppWrapper라는 컴포넌트에서 useLocation 훅을 사용
const AppWrapper = () => {
    const location = useLocation(); // useLocation을 여기서 사용

    return (
        <div className="App">
            <nav>
                <ul>
                    <li><Link to="/">기사 검색</Link></li>
                    <li><Link to="/mypage">내 성향 보기</Link></li>
                </ul>
            </nav>

            {/* Routes 설정 */}
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/mypage" element={<MyPage />} />
            </Routes>

            {/* 현재 경로가 '/'일 때만 SearchBar 렌더링 */}
            {location.pathname === '/' && <SearchBar />}
        </div>
    );
};

// Router를 최상위로 두고 AppWrapper로 감싼다
const App = () => {
    return (
        <Router>
            <AppWrapper />
        </Router>
    );
};

export default App;
