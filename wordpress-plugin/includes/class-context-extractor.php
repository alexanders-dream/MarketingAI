<?php

class Marketing_AI_Context_Extractor {

    public function extract_site_context() {
        $context = [
            'site_name' => get_bloginfo('name'),
            'site_description' => get_bloginfo('description'),
            'site_url' => get_site_url(),
            'recent_posts' => $this->get_recent_posts(10),
            'categories' => $this->get_terms('category'),
            'tags' => $this->get_terms('post_tag'),
            'theme' => wp_get_theme()->get('Name'),
            'webhook_url' => site_url('?mkai_hook=1'),
            'webhook_secret' => get_option('marketing_ai_webhook_secret')
        ];
        
        // WooCommerce extraction if available
        if (class_exists('WooCommerce')) {
            $context['products'] = $this->get_recent_products(10);
            $context['product_categories'] = $this->get_terms('product_cat');
        }
        
        // Extract recent brand images
        $img_extractor = new Marketing_AI_Image_Extractor();
        $context['brand_images'] = $img_extractor->get_all_brand_assets(10);
        
        return $context;
    }

    private function get_recent_posts($limit = 5) {
        $recent_posts = wp_get_recent_posts(['numberposts' => $limit, 'post_status' => 'publish']);
        $formatted = [];
        foreach($recent_posts as $post) {
            $formatted[] = [
                'title' => $post['post_title'],
                'url' => get_permalink($post['ID']),
                'excerpt' => wp_trim_words($post['post_content'], 20)
            ];
        }
        return $formatted;
    }
    
    private function get_recent_products($limit = 5) {
        $args = ['limit' => $limit, 'status' => 'publish', 'orderby' => 'date', 'order' => 'DESC'];
        $products = wc_get_products($args);
        $formatted = [];
        foreach($products as $p) {
            $formatted[] = [
                'name' => $p->get_name(),
                'price' => $p->get_price(),
                'url' => $p->get_permalink(),
                'short_description' => wp_trim_words($p->get_short_description(), 15)
            ];
        }
        return $formatted;
    }

    private function get_terms($taxonomy) {
        $terms = get_terms(['taxonomy' => $taxonomy, 'hide_empty' => true]);
        $formatted = [];
        if (!is_wp_error($terms)) {
            foreach($terms as $term) {
                $formatted[] = $term->name;
            }
        }
        return $formatted;
    }
}
