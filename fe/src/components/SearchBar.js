// fe/src/components/SearchBar.js

import React, { useState, useEffect } from 'react';
import { findSimilarArticles } from '../api';  // 새로운 API 호출 함수
import ArticleList from './ArticleList';

const SearchBar = () => {
    const [query, setQuery] = useState('');
    const [date, setDate] = useState(''); // 날짜 상태 (YYYYMMDD 형식)
    const [userId, setUserId] = useState('');
    const [articles, setArticles] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        console.log(articles);  // articles 배열이 제대로 업데이트되는지 확인
    }, [articles]);

    const handleSearch = async () => {
        setLoading(true);
        try {
            const results = await findSimilarArticles(date, query, userId);
            setArticles(results);
        } catch (error) {
            console.error('Error finding similar articles:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h1>필터버블 해소를 위한 뉴스레터</h1>
            <input
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="닉네임 입력"
            />
            <input
                type="number"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                placeholder="날짜 입력 (YYYYMMDD)"
            />
            <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="검색어 입력"
            />
            <button onClick={handleSearch} disabled={loading}>
                {loading ? '검색 중...' : '검색'}
            </button>
            <ArticleList articles={articles} />
        </div>
    );
};

export default SearchBar;
