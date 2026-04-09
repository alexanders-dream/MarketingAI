<?php

class Marketing_AI_Image_Extractor {
    public function handle_image_upload($image_url, $post_id) {
        require_once(ABSPATH . 'wp-admin/includes/media.php');
        require_once(ABSPATH . 'wp-admin/includes/file.php');
        require_once(ABSPATH . 'wp-admin/includes/image.php');

        $tmp = download_url($image_url);
        if (is_wp_error($tmp)) {
            return false;
        }

        $file_array = [
            'name' => basename(parse_url($image_url, PHP_URL_PATH)),
            'tmp_name' => $tmp
        ];

        $attach_id = media_handle_sideload($file_array, $post_id);
        if (is_wp_error($attach_id)) {
            @unlink($file_array['tmp_name']);
            return false;
        }

        return $attach_id;
    }
    
    public function get_all_brand_assets($limit = 10) {
        $args = [
            'post_type'      => 'attachment',
            'post_mime_type' => 'image',
            'post_status'    => 'inherit',
            'posts_per_page' => $limit,
            'orderby'        => 'date',
            'order'          => 'DESC'
        ];
        
        $images = get_posts($args);
        $assets = [];
        
        foreach($images as $img) {
            $assets[] = [
                'id' => $img->ID,
                'url' => wp_get_attachment_url($img->ID),
                'alt' => get_post_meta($img->ID, '_wp_attachment_image_alt', true),
                'title' => $img->post_title
            ];
        }
        
        return $assets;
    }
}
