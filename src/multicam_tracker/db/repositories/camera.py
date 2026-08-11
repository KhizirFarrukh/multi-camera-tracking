"""Postgres implementations of the camera and camera-link repositories."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from multicam_tracker.db.mappers import (
    camera_link_to_domain,
    camera_link_to_orm,
    camera_to_domain,
    camera_to_orm,
)
from multicam_tracker.db.orm import CameraLinkORM, CameraORM
from multicam_tracker.db.repositories._base import (
    PostgresRepositoryBase,
    rowcount_of,
    storage_errors,
)
from multicam_tracker.models import Camera, CameraLink

__all__ = ["PostgresCameraLinkRepository", "PostgresCameraRepository"]


class PostgresCameraRepository(PostgresRepositoryBase):
    """Camera persistence backed by Postgres."""

    def upsert(self, camera: Camera) -> Camera:
        """Insert or replace a camera. See the protocol for full semantics.

        Args:
            camera: The camera to store.

        Returns:
            The stored camera.

        Raises:
            StorageError: If the write fails.
        """
        row = camera_to_orm(camera)
        values = {column.key: getattr(row, column.key) for column in CameraORM.__table__.columns}
        statement = pg_insert(CameraORM).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[CameraORM.camera_id],
            set_={key: value for key, value in values.items() if key != "camera_id"},
        )

        with storage_errors("camera upsert", camera_id=camera.camera_id):
            self._session.execute(statement)
            self._session.flush()
        return camera

    def get(self, camera_id: str) -> Camera | None:
        """Return one camera by id.

        Args:
            camera_id: The identifier to look up.

        Returns:
            The camera, or ``None``.

        Raises:
            StorageError: If the read fails.
        """
        with storage_errors("camera lookup", camera_id=camera_id):
            row = self._session.get(CameraORM, camera_id)
        return camera_to_domain(row) if row is not None else None

    def list_enabled(self) -> list[Camera]:
        """Return every enabled camera, ordered by id.

        Returns:
            The enabled cameras.

        Raises:
            StorageError: If the read fails.
        """
        statement = (
            select(CameraORM).where(CameraORM.enabled.is_(True)).order_by(CameraORM.camera_id)
        )
        with storage_errors("enabled camera listing"):
            rows = self._session.execute(statement).scalars().all()
        return [camera_to_domain(row) for row in rows]

    def delete(self, camera_id: str) -> bool:
        """Delete one camera.

        Args:
            camera_id: The identifier to delete.

        Returns:
            ``True`` if a row was removed.

        Raises:
            StorageError: If the camera still has sightings (ON DELETE RESTRICT)
                or the write fails.
        """
        statement = delete(CameraORM).where(CameraORM.camera_id == camera_id)
        with storage_errors("camera delete", camera_id=camera_id):
            result = self._session.execute(statement)
            self._session.flush()
        return rowcount_of(result) > 0


class PostgresCameraLinkRepository(PostgresRepositoryBase):
    """Camera-link persistence backed by Postgres."""

    def upsert(self, link: CameraLink) -> CameraLink:
        """Insert or replace the link for one ordered pair.

        Args:
            link: The link to store.

        Returns:
            The stored link.

        Raises:
            StorageError: If an endpoint is unknown or the write fails.
        """
        row = camera_link_to_orm(link)
        values = {
            column.key: getattr(row, column.key) for column in CameraLinkORM.__table__.columns
        }
        keys = {"from_camera_id", "to_camera_id"}
        statement = pg_insert(CameraLinkORM).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[CameraLinkORM.from_camera_id, CameraLinkORM.to_camera_id],
            set_={key: value for key, value in values.items() if key not in keys},
        )

        with storage_errors(
            "camera link upsert",
            from_camera_id=link.from_camera_id,
            to_camera_id=link.to_camera_id,
        ):
            self._session.execute(statement)
            self._session.flush()
        return link

    def get_link(self, from_camera_id: str, to_camera_id: str) -> CameraLink | None:
        """Return the link declared for one ordered pair.

        Args:
            from_camera_id: Origin camera.
            to_camera_id: Destination camera.

        Returns:
            The link, or ``None``.

        Raises:
            StorageError: If the read fails.
        """
        with storage_errors(
            "camera link lookup", from_camera_id=from_camera_id, to_camera_id=to_camera_id
        ):
            row = self._session.get(CameraLinkORM, (from_camera_id, to_camera_id))
        return camera_link_to_domain(row) if row is not None else None

    def get_links_from(self, camera_id: str) -> list[CameraLink]:
        """Return links whose origin is ``camera_id``.

        Args:
            camera_id: Origin camera.

        Returns:
            The links, ordered by destination.

        Raises:
            StorageError: If the read fails.
        """
        statement = (
            select(CameraLinkORM)
            .where(CameraLinkORM.from_camera_id == camera_id)
            .order_by(CameraLinkORM.to_camera_id)
        )
        with storage_errors("camera link listing", camera_id=camera_id):
            rows = self._session.execute(statement).scalars().all()
        return [camera_link_to_domain(row) for row in rows]

    def list_all(self) -> list[CameraLink]:
        """Return every link, ordered by origin then destination.

        Returns:
            All declared links.

        Raises:
            StorageError: If the read fails.
        """
        statement = select(CameraLinkORM).order_by(
            CameraLinkORM.from_camera_id, CameraLinkORM.to_camera_id
        )
        with storage_errors("camera link listing"):
            rows = self._session.execute(statement).scalars().all()
        return [camera_link_to_domain(row) for row in rows]
