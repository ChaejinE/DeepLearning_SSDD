from tf_2.segmentation.ssdd.train.loss_ops import segmentation_loss, classification_loss
import tensorflow as tf


def seg_train_fn(strategy, model, optimizer, weight, batch_size):
    with strategy.scope():
        def step_fn(inputs, labels, global_batch_size):
            with tf.GradientTape() as tape:
                feature_maps, masks = model(inputs)
                loss = segmentation_loss(labels, masks, weight) / global_batch_size

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            tf.debugging.check_numerics(feature_maps, 'feature_maps nan', name='feature_maps_check')
            tf.debugging.check_numerics(masks, 'masks nan', name='masks_check')
            tf.debugging.check_numerics(loss, 'seg_loss nan', name='seg_loss_check')

            return loss

        @tf.function
        def distributed_fn(inputs, labels):
            per_replica_losses = strategy.experimental_run_v2(step_fn,
                                                              args=(inputs, labels, batch_size,))

            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    return distributed_fn


def seg_valid_fn(strategy, model, weight, batch_size):
    with strategy.scope():
        def step_fn(inputs, labels, global_batch_size):
            feature_maps, masks = model(inputs)
            loss = segmentation_loss(labels, masks, weight) / global_batch_size

            tf.debugging.check_numerics(feature_maps, 'feature_maps nan', name='feature_maps_check')
            tf.debugging.check_numerics(masks, 'masks nan', name='masks_check')
            tf.debugging.check_numerics(loss, 'seg_loss nan', name='seg_loss_check')

            return loss

        @tf.function
        def distributed_fn(inputs, labels):
            per_replica_losses = strategy.experimental_run_v2(step_fn, args=(inputs, labels, batch_size,))

            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    return distributed_fn


def dec_train_fn(strategy, segnet, decnet, optimizer, batch_size):
    with strategy.scope():
        def step_fn(inputs, labels, global_batch_size):
            """
            :param inputs: tensor of shape (batch size, h, w, ch)
            :param labels: tensor of shape (batch size, ) with value (0 or 1)
            :param global_batch_size: batch size for Strategy
            :param optimizer: sgd
            :return:
            """
            feature_maps, masks = segnet(inputs)
            with tf.GradientTape() as tape:
                cls_logits = decnet([masks, feature_maps])
                loss = classification_loss(labels, cls_logits) / global_batch_size
            gradients = tape.gradient(loss, decnet.trainable_variables)
            optimizer.apply_gradients(zip(gradients, decnet.trainable_variables))

            tf.debugging.check_numerics(cls_logits, 'cls_logits nan', name='cls_logits_check')

            return loss

        @tf.function
        def distributed_fn(inputs, labels):
            per_replica_losses = strategy.experimental_run_v2(step_fn,
                                                              args=(inputs, labels, batch_size,))

            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    return distributed_fn


def dec_valid_fn(strategy, segnet, decnet, batch_size):
    with strategy.scope():
        def step_fn(inputs, labels, global_batch_size):
            feature_maps, masks = segnet(inputs)
            cls_logits = decnet([masks, feature_maps])
            loss = classification_loss(labels, cls_logits) / global_batch_size

            tf.debugging.check_numerics(cls_logits, 'cls_logits nan', name='cls_logits_check')

            return loss

        @tf.function
        def distributed_fn(inputs, labels):
            per_replica_losses = strategy.experimental_run_v2(step_fn, args=(inputs, labels, batch_size,))

            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    return distributed_fn


def segdec_train_fn(strategy, segnet, decnet, optimizer, weight, batch_size):
    with strategy.scope():
        def step_fn(inputs, mask_labels, cls_labels, global_batch_size):
            with tf.GradientTape() as tape:
                feature_maps, masks = segnet(inputs)
                cls_logits = decnet([masks, feature_maps])
                seg_loss = segmentation_loss(mask_labels, masks, weight)
                dec_loss = classification_loss(cls_labels, cls_logits)
                total_loss = (seg_loss + dec_loss) / global_batch_size
            variables = segnet.trainable_variables + decnet.trainable_variables
            gradients = tape.gradient(total_loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            tf.debugging.check_numerics(cls_logits, 'cls_logits nan', name='cls_logits_check')

            return total_loss

        @tf.function
        def distributed_fn(inputs, labels):
            mask_labels, cls_labels = labels
            per_replica_losses = strategy.experimental_run_v2(step_fn, args=(inputs, mask_labels,
                                                                             cls_labels, batch_size,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    return distributed_fn


def segdec_valid_fn(strategy, segnet, decnet, weight, batch_size):
    with strategy.scope():
        def step_fn(inputs, mask_labels, cls_labels, global_batch_size):
            feature_maps, masks = segnet(inputs)
            cls_logits = decnet([masks, feature_maps])
            seg_loss = segmentation_loss(mask_labels, masks, weight)
            dec_loss = classification_loss(cls_labels, cls_logits)
            total_loss = (seg_loss + dec_loss) / global_batch_size

            tf.debugging.check_numerics(cls_logits, 'cls_logits nan', name='cls_logits_check')

            return total_loss

        @tf.function
        def distributed_fn(inputs, labels):
            mask_labels, cls_labels = labels
            per_replica_losses = strategy.experimental_run_v2(step_fn, args=(inputs, mask_labels,
                                                                             cls_labels, batch_size,))

            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    return distributed_fn
