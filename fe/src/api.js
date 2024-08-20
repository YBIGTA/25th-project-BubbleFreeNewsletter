// fe/src/api.js
import axios from 'axios';

// FastAPI 서버 주소
const API_URL = 'http://localhost:8000';

export const findSimilarArticles = async (date, query, userId) => {
    try {
        const response = await axios.get(`${API_URL}/search/`, {
            params: { date, query, user_id: userId }
        });
        return response.data;
    } catch (error) {
        console.error('유사한 기사 찾기 오류:', error);
        throw error;
    }
};


export const getArticle = async (articleId) => {
    try {
        const response = await axios.get(`${API_URL}/article/infer/${articleId}`);
        return response.data;
    } catch (error) {
        console.error('Error fetching article by ID:', error);
        throw error;
    }
};