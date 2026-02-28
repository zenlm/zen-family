import { defineConfig } from 'fumadocs-mdx/config';
import { z } from 'zod';

export default defineConfig({
  name: 'Zen AI Documentation',
  description: 'Ultra-efficient language models for edge deployment',
  
  // Documentation structure
  sidebar: {
    '/': [
      {
        title: 'Getting Started',
        items: [
          { title: 'Introduction', href: '/' },
          { title: 'Quick Start', href: '/quickstart' },
          { title: 'Installation', href: '/installation' },
          { title: 'v1.0.1 Updates', href: '/updates/v1.0.1' },
        ],
      },
      {
        title: 'Models',
        items: [
          { title: 'Overview', href: '/models' },
          { title: 'Zen-Nano (600M)', href: '/models/zen-nano' },
          { title: 'Zen-Eco (4B)', href: '/models/zen-eco' },
          { title: 'Zen-Coder (480B-A35B)', href: '/models/zen-coder' },
          { title: 'Zen-Omni (30B-A3B)', href: '/models/zen-omni' },
          { title: 'Zen-Next (80B-A3B)', href: '/models/zen-next' },
        ],
      },
      {
        title: 'Training',
        items: [
          { title: 'Zoo-Gym Framework', href: '/training/zoo-gym' },
          { title: 'Fine-tuning Guide', href: '/training/fine-tuning' },
          { title: 'LoRA Configuration', href: '/training/lora' },
          { title: 'Recursive Improvement', href: '/training/recursive' },
          { title: 'MoE Training', href: '/training/moe' },
        ],
      },
      {
        title: 'Architecture',
        items: [
          { title: 'Technical Overview', href: '/architecture' },
          { title: 'Dense Models', href: '/architecture/dense' },
          { title: 'MoE Models', href: '/architecture/moe' },
          { title: 'Multimodal', href: '/architecture/multimodal' },
          { title: 'Ultra-Sparse', href: '/architecture/ultra-sparse' },
        ],
      },
      {
        title: 'Deployment',
        items: [
          { title: 'Deployment Guide', href: '/deployment' },
          { title: 'Edge Deployment', href: '/deployment/edge' },
          { title: 'Server Deployment', href: '/deployment/server' },
          { title: 'Cloud Deployment', href: '/deployment/cloud' },
          { title: 'Format Conversion', href: '/deployment/conversion' },
        ],
      },
      {
        title: 'API Reference',
        items: [
          { title: 'Python API', href: '/api/python' },
          { title: 'REST API', href: '/api/rest' },
          { title: 'Model Configs', href: '/api/configs' },
          { title: 'Zoo-Gym API', href: '/api/zoo-gym' },
        ],
      },
      {
        title: 'Papers',
        items: [
          { title: 'Whitepaper', href: '/papers/whitepaper' },
          { title: 'Technical Paper', href: '/papers/technical' },
          { title: 'o1 Reasoning', href: '/papers/o1-reasoning' },
          { title: 'Architecture Ref', href: '/papers/architecture' },
        ],
      },
      {
        title: 'Resources',
        items: [
          { title: 'Benchmarks', href: '/resources/benchmarks' },
          { title: 'Model Cards', href: '/resources/model-cards' },
          { title: 'FAQ', href: '/resources/faq' },
          { title: 'Community', href: '/resources/community' },
        ],
      },
    ],
  },
  
  // Theme configuration
  theme: {
    accentColor: 'blue',
    mode: 'auto',
    logo: '/logo.svg',
  },
  
  // Search configuration
  search: {
    enabled: true,
    algolia: {
      appId: process.env.ALGOLIA_APP_ID,
      apiKey: process.env.ALGOLIA_API_KEY,
      indexName: 'zen-docs',
    },
  },
  
  // Social links
  social: {
    github: 'https://github.com/zenlm',
    discord: 'https://discord.gg/zen-ai',
    twitter: 'https://twitter.com/zenai',
  },
  
  // Analytics
  analytics: {
    google: process.env.GA_MEASUREMENT_ID,
  },
  
  // Custom metadata
  metadata: {
    title: 'Zen AI Documentation',
    description: 'Comprehensive documentation for Zen AI models',
    keywords: [
      'zen ai',
      'language models',
      'edge ai',
      'moe models',
      'zoo-gym',
      'qwen3',
      'hanzo ai',
      'zoo labs',
    ],
    authors: [
      { name: 'Hanzo AI', url: 'https://hanzo.ai' },
      { name: 'Zoo Labs Foundation', url: 'https://zoo.ai' },
    ],
  },
  
  // Export configuration
  export: {
    output: 'static',
    trailingSlash: true,
  },
});