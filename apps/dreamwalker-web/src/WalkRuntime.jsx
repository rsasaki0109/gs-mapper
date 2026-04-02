import { Entity, Gltf, Modify } from '@playcanvas/react';
import { Camera, Collision, RigidBody, Script } from '@playcanvas/react/components';
import { useModel, usePhysics } from '@playcanvas/react/hooks';
import { FirstPersonController } from 'playcanvas/scripts/esm/first-person-controller.mjs';
import { useEffect } from 'react';

function WalkProxyColliders({ walkProxyColliders, walkController }) {
  const { isPhysicsLoaded } = usePhysics();

  if (!isPhysicsLoaded) {
    return null;
  }

  return walkProxyColliders.map((collider) => (
    <Entity key={collider.id} name={collider.id} position={collider.position}>
      <Collision type="box" halfExtents={collider.halfExtents} />
      <RigidBody type="static" restitution={0} friction={walkController.friction} />
    </Entity>
  ));
}

function WalkRigController({
  playerControllerRef,
  walkCameraEntity,
  walkController
}) {
  const { isPhysicsLoaded } = usePhysics();

  if (!isPhysicsLoaded || !walkCameraEntity) {
    return null;
  }

  return (
    <Script
      ref={playerControllerRef}
      script={FirstPersonController}
      camera={walkCameraEntity}
      lookSens={walkController.lookSensitivity}
      speedGround={walkController.speedGround}
      speedAir={walkController.speedAir}
      sprintMult={walkController.sprintMultiplier}
      jumpForce={walkController.jumpForce}
    />
  );
}

function ColliderMeshLayer({
  colliderMeshUrl,
  showColliderDebug,
  walkController,
  onStatusChange
}) {
  const { isPhysicsLoaded } = usePhysics();
  const { asset, loading, error } = useModel(colliderMeshUrl);

  useEffect(() => {
    if (!isPhysicsLoaded) {
      onStatusChange?.({
        mode: 'loading',
        error: null
      });
      return;
    }

    if (error) {
      onStatusChange?.({
        mode: 'error',
        error
      });
      return;
    }

    if (loading || !asset) {
      onStatusChange?.({
        mode: 'loading',
        error: null
      });
      return;
    }

    onStatusChange?.({
      mode: 'mesh',
      error: null
    });
  }, [asset, error, isPhysicsLoaded, loading, onStatusChange]);

  if (!isPhysicsLoaded || !asset) {
    return null;
  }

  return (
    <Entity name="ColliderMeshRoot">
      <Gltf asset={asset} key={`${asset.id}-${showColliderDebug ? 'debug' : 'hidden'}`} render={showColliderDebug}>
        <Modify.Node path="**[render]">
          <Collision type="mesh" />
          <RigidBody type="static" friction={walkController.friction} restitution={0} />
        </Modify.Node>
      </Gltf>
    </Entity>
  );
}

export default function WalkRuntime({
  currentPreset,
  playerEntityRef,
  playerControllerRef,
  handleWalkCameraRef,
  walkCameraEntity,
  walkController,
  walkProxyColliders,
  colliderMeshUrl,
  showColliderDebug,
  shouldUseProxyColliders,
  onStatusChange
}) {
  return (
    <>
      <Entity
        name="WalkRig"
        ref={playerEntityRef}
        position={[
          currentPreset.position[0],
          currentPreset.position[1] - walkController.cameraLocalHeight,
          currentPreset.position[2]
        ]}>
        <Collision
          type="capsule"
          radius={walkController.capsuleRadius}
          height={walkController.capsuleHeight}
        />
        <RigidBody
          type="dynamic"
          mass={walkController.mass}
          friction={walkController.friction}
          restitution={0}
          linearDamping={0}
          angularDamping={0}
          linearFactor={[1, 1, 1]}
          angularFactor={[0, 0, 0]}
        />
        <WalkRigController
          playerControllerRef={playerControllerRef}
          walkCameraEntity={walkCameraEntity}
          walkController={walkController}
        />
        <Entity
          name="WalkCamera"
          ref={handleWalkCameraRef}
          position={[0, walkController.cameraLocalHeight, 0]}>
          <Camera enabled />
        </Entity>
      </Entity>

      {colliderMeshUrl ? (
        <ColliderMeshLayer
          colliderMeshUrl={colliderMeshUrl}
          showColliderDebug={showColliderDebug}
          walkController={walkController}
          onStatusChange={onStatusChange}
        />
      ) : null}
      {shouldUseProxyColliders ? (
        <WalkProxyColliders
          walkProxyColliders={walkProxyColliders}
          walkController={walkController}
        />
      ) : null}
    </>
  );
}
